Nustatyti: learningRate, gamma, epsilon, maxEpsilon, minEpsilon, decayRate, totalEpisodes, maxSteps

aplinka.generuotiŽemėlapį(dydis, tikimybės):
    žemėlapis = atsitiktinai užpildyti laukelius su "-", "T", "N" pagal  tikimybes
                bei nustatyti "S" ir "L"
    jumpers = atsitiktinai priskirti visas korteles kitai kažkuriai vienai kortelei


for episode in totalEpisodes:
    aplinka.reset()
    for step < max_steps:
        generuoti atsitiktinisSkaičius
        if atsitiktinisSkaičius > epsilon:
            veiksmas = geriausias iš qtable[dabartinėBūsena,]
        else:
            veiksmas = atsitiktinis iš aplinka.veiksmųAibė
        naujaBūsena, apdovanojimas = aplinka.žingsnis(veiksmas):
            if veiksmas == LEFT:
                judinti agentą kairėn
            elif veiksmas == DOWN:
                judinti agentą žemyn
            elif veiksmas == RIGHT:
                judinti agentą dešinėn
            elif veiksmas == UP:
                judinti agentą viršun
            elif veiksmas == META:
                judėti į kortelę pagal sugeneruotą jumpers sąrašą. 
        taikyti Q naudos funkciją ir gauti naują Q letelės vertę
        qtable[dabartinėBūsena, veiksmas] = qtable[dabartinėBūsena, veiksmas] + learningRate * (apdovanojimas + gamma * geriausias iš qtable[naujaBūsena,]) - qtable[dabartinėBūsena, veiksmas])

        if tikslas pasiektas :
            nutraukti epizodą
    
    Mažinti epsilon vertę, kad agentas darytų mažiau atsitiktinių veiksmų 
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
