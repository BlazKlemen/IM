' ======================================================================
' GLOBALNE SPREMENLJIVKE (za komunikacijo med nitmi)
' ======================================================================
Global Boolean g_AbortFlag               ' Zastavica za sporočanje napake iz MovementMonitor-ja
Global Boolean g_MovementMonitorRunning  ' Zastavica za izklop vzporedne niti

Function main
    ' Deklaracija spremenljivk (usklajena s Pythonom)
    Real dx, dy, dz, rx, ry, rz
    Int32 type
    String line$
    String command$(7) ' Polje za razclenjevanje prejetih podatkov
    Integer i
    
    ' Spremenljivke za TCP kalibracijo (Koda 3)
    Real cx, cy, cz, cu, cv, cw
    String msg$
    
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
        WaitNet #201
        
        Print "--- Povezava vzpostavljena! Zacenjam sekvenco... ---"
        
        Do
            If ChkNet(201) > 0 Then
                
                Input #201, line$
                ParseStr line$, command$(), " "
                type = Val(command$(0)) ' Prvi parameter je vedno KODA UKAZA
                
                ' --- KODA 2: Napaka / Premik izdelka (Abort) pred zacetkom ---
                If type = 2 Then
                    Print "[Robot] Kamera javlja PREMIK IZDELKA pred začetkom!"
                    Exit Do ' Izstopimo iz notranje zanke
                EndIf
                
                ' --- KODA 3: Kalibracija TCP (4 tocke) ---
                If type = 3 Then
                    Print "=== ZACETEK TCP KALIBRACIJE (4 TOCKO VNOS) ==="
                    
                    For i = 1 To 4
                        Print "Zapogaj robota do konice pod kotom ", i, " in pritisni ENTER v Epson konzoli..."
                        
                        ' Program se ustavi in caka na pritisk tipke Enter
                        Input line$ 
                        
                        ' Preklopimo na Tool 0, saj zelimo izmeriti tocno pozicijo flange!
                        Tool 0
                        cx = CX(CurPos)
                        cy = CY(CurPos)
                        cz = CZ(CurPos)
                        cu = CU(CurPos)
                        cv = CV(CurPos)
                        cw = CW(CurPos)
                        
                        ' Sestavimo niz v formatu "x,y,z,u,v,w"
                        msg$ = Str$(cx) + "," + Str$(cy) + "," + Str$(cz) + "," + Str$(cu) + "," + Str$(cv) + "," + Str$(cw)
                        
                        ' Posljemo tocko Pythonu preko socketa
                        Write #201, msg$
                        Print "[Robot] Tocka ", i, " poslana Pythonu: ", msg$
                    Next
                    
                    Print "=== TCP KALIBRACIJA NA ROBOTU ZAKLJUCENA ==="
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
                    
                    ' Nastavimo nagnjeno mizo / kos
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
                    For i = 0 To 4
                        If g_AbortFlag Then
                            Exit For ' Če je MovementMonitor ustavil robota z AbortMotion, izstopimo
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
                        Write #201, "1"
                        Print "[Robot] Nanos uspešno zaključen!"
                    EndIf
                EndIf
                
            EndIf
            
            Wait 0.02 ' Kratka pavza za razbremenitev krmilnika (20 ms)
        Loop ' Konec notranje zanke (Loop za tocke)
        
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