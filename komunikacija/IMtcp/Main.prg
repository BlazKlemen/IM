' ======================================================================
' GLOBALNE SPREMENLJIVKE (za komunikacijo med nitmi)
' ======================================================================
Global Boolean g_AbortFlag               ' Zastavica, ki sporoči main-u, da je kamera javila napako
Global Boolean g_MovementMonitorRunning  ' Zastavica za varen izklop vzporedne niti

Function main
    ' Deklaracija lokalnih spremenljivk
    Real dx, dy, dz, rx, ry, rz
    Int32 type
    String line$
    String command$(7)
    Integer i
    
    ' Zacetne nastavitve robota
    Motor On
    Power High
    Speed 50
    Accel 10, 10
    SpeedS 500
    AccelS 300, 300
    
    Go P0 +Z(30)
    
    Do
        Print "--- Cakam na povezavo s Pythonom na portu 12345... ---"
        CloseNet #201
        Wait 0.5
      
        OpenNet #201 As Client
        Wait 0.5
        WaitNet #201
        
        Print "--- Povezava vzpostavljena! Zacenjam sekvenco... ---"
        
        Do
            If ChkNet(201) > 0 Then
                
                Input #201, line$
                ParseStr line$, command$(), " "
                type = Val(command$(0)) ' Prvi parameter je vedno KODA UKAZA
                
                ' --- KODA 2: Napaka pred zacetkom ---
                If type = 2 Then
                    Print "[Robot] Kamera javlja PREMIK IZDELKA pred začetkom!"
                    Exit Do
                EndIf
                
                ' --- KODA 1: Prejet nov koordinatni sistem za zacetek cikla ---
                If type = 1 Then
                    
                    dx = Val(command$(2))
                    dy = Val(command$(3))
                    dz = Val(command$(4))
                    rx = Val(command$(5))
                    ry = Val(command$(6))
                    rz = Val(command$(7))
                    
                    Print "Prejel koordinatni sistem. Nastavljam Local 1..."
                    Local 1, XY(dx, dy, dz, rx, ry, rz)
                    
                    ' Dvignemo se nad zacetno tocko P0 glede na Local 1
                    Move P0 +Z(50) /1
                    Wait 0.5 ' Stabilizacija
                    
                    ' --------------------------------------------------
                    ' 1. ZAGON VZPOREDNE NITI MovementMonitor
                    ' --------------------------------------------------
                    g_AbortFlag = False
                    g_MovementMonitorRunning = True
                    Xqt MovementMonitor, NoPause ' Poženemo nadzor premika v ozadju!
                    
                    Print "[Robot] Začenjam nanos P0 -> P4 s stalnim nadzorom kamere..."
                    
                    ' ODREMO POT OD P0 DO P4
                    AutoLJM On
                    For i = 0 To 4
                        If g_AbortFlag Then
                            Exit For ' Če je MovementMonitor ustavil robota, takoj izstopimo
                        EndIf
                        
                        Print "Premik na tocko P", i
                        Move P(i) /1
                    Next
                    
                    ' --------------------------------------------------
                    ' 2. ZAKLJUČEK VZPOREDNE NITI
                    ' --------------------------------------------------
                    g_MovementMonitorRunning = False
                    Wait 0.05
                    Quit MovementMonitor ' Varno ugasnemo nit
                    
                    ' --------------------------------------------------
                    ' 3. PREVERJANJE REZULTATA NANOSA
                    ' --------------------------------------------------
                    If g_AbortFlag Then
                        Print "[Robot] CIKEL PREKINJEN zaradi premika kosa! Umikam robota..."
                        Move P0 +Z(50) ' Varen umik navzgor
                        Exit Do        ' Prekinemo notranjo zanko
                    Else
                        ' Uspeh: Pošljemo Pythonu potrditev (Koda 1)
                        Write #201, "1"
                        Print "[Robot] Nanos uspešno zaključen!"
                    EndIf
                EndIf
                
            EndIf
            
            Wait 0.02 ' Kratka pavza za razbremenitev krmilnika (20 ms)
        Loop ' Konec notranje zanke
        
        CloseNet #201
        Print "--- Povezava zaprta. Pripravljen na nov cikel. ---"
        Wait 1.0
        
    Loop

Fend

' ======================================================================
' VZPOREDNA NIT: Spremlja omrežni port za premik kosa med delom
' ======================================================================
Function MovementMonitor
    String line$
    String command$(7)
    Int32 type
    
    Do While g_MovementMonitorRunning
        ' Preverimo, če je Python poslal nov podatek preko socketa
        If ChkNet(201) > 0 Then
            Input #201, line$
            ParseStr line$, command$(), " "
            type = Val(command$(0))
            
            ' Če prejmemo kodo 2 (Premik kosa), TAKOJ ustavimo robota!
            If type = 2 Then
                Print "[MovementMonitor] ZAZNAN PREMIK IZDELKA (type=2)! Ustavljam robota!"
                g_AbortFlag = True
                
                ' Ukaz AbortMotion v trenutku prekine fizično gibanje robota
                AbortMotion
                Exit Do
            EndIf
        EndIf
        
        Wait 0.01 ' Hitro osveževanje (vsakih 10 ms)
    Loop
Fend