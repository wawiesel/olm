#
# support tools for automated ORIGEN library information collection
#
#   by BENJAMIN R. BETZLER
#      on 20 May 2014
#

import numpy as np


# driver for extracting relevant information from input files
def extractScaleInput(inputFile):
    # get a cleaned input file string and check for SAS input
    fStream = cleanScaleInput(inputFile)
    if fStream == True:
        return True, [[[]]], True, 0

    # cut out and interpret the alias section if it exists
    if fStream.find("read alias") != -1:
        alStream = fStream[fStream.find("read alias") + 10 : fStream.rfind("end alias")]
        listAliases = interperetAlias(alStream)
    else:
        listAliases = []

    # cut out and interpret composition input section
    compStream = fStream[fStream.find("read comp") + 9 : fStream.rfind("end comp")]
    compIDs, compNames = interperetCompositions(compStream, listAliases)

    # cut out and interpret CellData input section
    cdStream = fStream[fStream.find("celldata") + 8 : fStream.rfind("end celldata")]
    parTypes, parValues, parComps, aliasedSets = interperetCellData(
        cdStream, compIDs, compNames
    )

    # record the number of aliased calculations
    aliasedCalcs = 0
    for lists in listAliases:
        for strings in aliasedSets:
            if lists[0] == strings:
                aliasedCalcs = aliasedCalcs + len(lists) - 2

    # for LaTeX reference
    refName = inputFile.split("/")[-1].split(".")[0]
    assignParameter("Name", refName, parTypes, parValues, 0, 0)

    return parTypes, parValues, parComps, aliasedCalcs


# returns a long string containing the entire input file
def cleanScaleInput(inputFile):
    # read in the file line by line
    fileLines = open(inputFile).readlines()
    fileStream = ""

    # remove comments and convert to string
    for strings in fileLines:
        if strings[0] != "'":
            fileStream = fileStream + strings

    # remove control characters and format equal signs and read statements
    fileStream = fileStream.replace("\n", "  ")
    fileStream = fileStream.replace("\r", "  ")
    fileStream = fileStream.replace("read composition", "read comp")
    while fileStream.find("= ") != -1:
        fileStream = fileStream.replace("= ", "=")
    while fileStream.find(" =") != -1:
        fileStream = fileStream.replace(" =", "=")

    if (fileStream.find("=sas2") != -1) or (fileStream.find("csas2") != -1):
        fileStream = True

    return fileStream


# returns aliased set for the compositions from the 'read alias' input
def interperetAlias(aliasStream):
    # initializations
    listAliases = [[]]
    tempAliases = []
    needAlias = True
    needCompID = False

    # walk through each word of the input
    aliasWords = aliasStream.split()
    for strings in aliasWords:
        if needCompID & (strings != "end"):
            if strings.find("-") != -1:
                tempRange = [int(strings.split("-")[0]), int(strings.split("-")[1])]
                for i in range(tempRange[1] - tempRange[0] + 1):
                    tempAliases.append(str(tempRange[0] + i))
            else:
                tempAliases.append(strings)
        if needAlias:
            tempAliases.append(strings)
            needAlias = False
            needCompID = True
        if strings == "end":
            needAlias = True
            needCompID = False
            listAliases.append(tempAliases)
            tempAliases = []

    listAliases = listAliases[1 : len(listAliases)]

    return listAliases


# returns material names from the 'read comp' input
def interperetCompositions(compositionStream, listAliases):
    # initializations
    listNames = []
    listIDs = []
    needName = True
    needID = False

    # walk through each word of the input
    compositionWords = compositionStream.split()
    for strings in compositionWords:
        if strings[0:4] == "arbm":
            needID = False
            needName = False
        if needID:
            listIDs.append(strings)
            needID = False
        if needName:
            listNames.append(strings)
            needName = False
            needID = True
        if strings == "end":
            needName = True
            needID = False

    # for an arbitrary material declaration
    for i in range(len(compositionWords)):
        if compositionWords[i][0:4] == "arbm":
            listNames.append(compositionWords[i])
            compIDLocation = i + 6 + 2 * int(float(compositionWords[i + 2]))
            listIDs.append(compositionWords[compIDLocation])

    # pair compositions with multiple constituent/materials
    longNames = []
    compositionIDs = []
    for i in range(len(listIDs)):
        storedName = False
        for j in range(len(compositionIDs)):
            if listIDs[i] == compositionIDs[j]:
                longNames[j] = longNames[j] + "/" + listNames[i]
                storedName = True
                break
        if not storedName:
            compositionIDs.append(listIDs[i])
            longNames.append(listNames[i])

    # add alias compositions to the end of the string
    for lists in listAliases:
        tempAlias = lists[0]
        if tempAlias in compositionIDs:
            for i in range(len(lists) - 1):
                compositionIDs.append(lists[i + 1])
                longNames.append(longNames[compositionIDs.index(tempAlias)])
        else:
            compositionIDs.append(tempAlias)
            longNames.append(longNames[compositionIDs.index(lists[1])])

    # hem very long compositions to list only four constituent/materials
    maxLength = 3
    compositionNames = []
    for strings in longNames:
        if strings.count("/") > maxLength:
            tempList = strings.split("/")
            tempString = ""
            for i in range(maxLength):
                tempString = tempString + tempList[i] + "/"
            tempString = tempString + "etc."
            compositionNames.append(tempString)
        else:
            compositionNames.append(strings)

    return compositionIDs, compositionNames


# returns relevant material and geometric information from the 'celldata' input
def interperetCellData(cellData, compositionIDs, compositionNames):
    # count the number of CellData sub sections and split into those blocks
    nLattices = cellData.count("latticecell")
    nMultiregions = cellData.count("multiregion")
    nDoubleHets = cellData.count("doublehet")
    if nLattices == 0:  # no LatticeCell blocks
        if nMultiregions == 0:  # no Multiregion blocks
            if nDoubleHets == 0:  # no DoubleHet blocks
                return True, [[[]]], True, [""]
    nTot = nLattices + nMultiregions

    # in cases with dancoff factors via the centrmData block
    nDancoffFactors = cellData.count("dan2pitch")
    for i in range(nDancoffFactors):
        centrmStream = cellData[cellData.find("centrm") : cellData.find("end centrm")]
        cellData = cellData.replace(centrmStream + "end centrmdata", " ")
    # in cases with multiregion blocks
    for i in range(nMultiregions):
        subStream = cellData[cellData.find("multiregion") : cellData.find("end zone")]
        cellData = cellData.replace(subStream, subStream.replace("end", " "))
        cellData = cellData.replace("multiregion", "Multiregion", 1)
        cellData = cellData.replace("end zone", "end ", 1)
    cellData = cellData[cellData.find("latticecell") : len(cellData)]
    blocks = cellData.split("end")

    # initializations and name assign
    parameterTypes = [["-"] * 20 for i in range(nTot)]
    parameterValues = [["-"] * 20 for i in range(nTot)]
    tempIDs = [["-"] * 20 for i in range(nTot)]
    parameterCompositions = [["-"] * 20 for i in range(nTot)]
    aliasedSets = [["-"] for i in range(nTot)]

    # extract information in CellData block
    for i in range(nTot):
        tempArray = np.array(blocks[i].split())
        assignParameter("Type", tempArray[1], parameterTypes, parameterValues, i, 1)
        assignParameter("Composition", "-", parameterTypes, parameterCompositions, i, 1)
        j = 2
        multiRegionEntry = False
        for k in range(len(tempArray)):
            if tempArray[k] == "latticecell":
                multiRegionEntry = False
            if tempArray[k] == "Multiregion":
                multiRegionEntry = True
            if tempArray[k].find("$") != -1:
                aliasedSets[i] = tempArray[k]
            if tempArray[k].find("=") != -1:
                tempParameter = tempArray[k].split("=")
                assignParameter(
                    tempParameter[0],
                    tempParameter[1],
                    parameterTypes,
                    parameterValues,
                    i,
                    j,
                )
                assignParameter(
                    "Composition", tempArray[k + 1], parameterTypes, tempIDs, i, j
                )
                for n in range(len(compositionIDs)):  # ID numbers to compositions
                    if tempIDs[i][j] == compositionIDs[n]:
                        parameterCompositions[i][j] = compositionNames[n]
                        break
                j = j + 1
            if multiRegionEntry:
                if parameterTypes[i][j - 1].find("Region") == -1:
                    parameterCompositions[i][j - 1] = "-"
                if tempArray[k].isdigit():
                    assignParameter(
                        "Composition", tempArray[k], parameterTypes, tempIDs, i, j
                    )
                    assignParameter(
                        "Multiregion Cell",
                        tempArray[k + 1],
                        parameterTypes,
                        parameterValues,
                        i,
                        j,
                    )
                    for n in range(len(compositionIDs)):  # ID numbers to compositions
                        if tempIDs[i][j] == compositionIDs[n]:
                            parameterCompositions[i][j] = compositionNames[n]
                            break
                    j = j + 1
        parameterTypes[i][0] = j - 1

    return parameterTypes, parameterValues, parameterCompositions, aliasedSets


# assigns parameter values to the list of lists pValues
# this is essentially a glossary of LatticeCell block terms
def assignParameter(tipe, val, pTypes, pValues, n, p):
    if tipe == "Name":  # name for LaTeX reference
        pValues[n][p] = val
    if tipe == "Type":  # type of lattice/multiregion cells
        # for LatticeCell blocks
        if val == "squarepitch":
            latticeType = "square"
        if val == "asquarepitch":
            latticeType = "annular square"
        if val == "triangpitch":
            latticeType = "triangular"
        if val == "atriangpitch":
            latticeType = "annular triangular"
        if val == "sphsquarep":
            latticeType = "square sphere"
        if val == "asphsquarep":
            latticeType = "annular square sphere"
        if val == "sphtriangp":
            latticeType = "triangular sphere"
        if val == "asphtriangp":
            latticeType = "annular triangular sphere"
        if val == "symmslabcell":
            latticeType = "symmetric slab"
        if val == "asymmslabcell":
            latticeType = "annular symmetric slab"
        # for Multiregion blocks
        if val == "slab":
            latticeType = "multiregion slab"
        if val == "cylindrical":
            latticeType = "multiregion cyl."
        if val == "spherical":
            latticeType = "multiregion sphere"
        if val == "buckledslab":
            latticeType = "multiregion buckled slab"
        if val == "buckledcyl":
            latticeType = "multiregion buckled cyl."
        # for DoubleHet blocks
        pTypes[n][p] = "Lattice " + str(n + 1) + " type"
        pValues[n][p] = latticeType
    # for Multiregion cell information
    if tipe == "Multiregion Cell":
        if pTypes[n][p - 1].find("Region") == -1:
            pTypes[n][p] = "Region 1 radius"
        else:
            tempNextRegionNumber = int(float(pTypes[n][p - 1].split()[1]) + 1)
            pTypes[n][p] = "Region " + str(tempNextRegionNumber) + " radius"
        pValues[n][p] = val
    # for composition information
    if tipe == "Composition":
        pValues[n][p] = val
    # for LatticeCell blocks
    if tipe == "imodr":
        pTypes[n][p] = "Inner moderator radius"
        pValues[n][p] = float(val)
    if tipe == "imodd":
        pTypes[n][p] = "Inner moderator radius"
        pValues[n][p] = float(val) / 2
    if tipe == "icladr":
        pTypes[n][p] = "Inner clad radius"
        pValues[n][p] = float(val)
    if tipe == "icladd":
        pTypes[n][p] = "Inner clad radius"
        pValues[n][p] = float(val) / 2
    if tipe == "igapr":
        pTypes[n][p] = "Inner gap radius"
        pValues[n][p] = float(val)
    if tipe == "igapd":
        pTypes[n][p] = "Inner gap radius"
        pValues[n][p] = float(val) / 2
    if tipe == "fuelr":
        pTypes[n][p] = "Fuel radius"
        pValues[n][p] = float(val)
    if tipe == "fueld":
        pTypes[n][p] = "Fuel radius"
        pValues[n][p] = float(val) / 2
    if tipe == "gapr":
        pTypes[n][p] = "Gap radius"
        pValues[n][p] = float(val)
    if tipe == "gapd":
        pTypes[n][p] = "Gap radius"
        pValues[n][p] = float(val) / 2
    if tipe == "cladr":
        pTypes[n][p] = "Fuel clad radius"
        pValues[n][p] = float(val)
    if tipe == "cladd":
        pTypes[n][p] = "Fuel clad radius"
        pValues[n][p] = float(val) / 2
    if tipe == "hpitch":
        pTypes[n][p] = "Full pitch"
        pValues[n][p] = 2 * float(val)
    if tipe == "pitch":
        pTypes[n][p] = "Full pitch"
        pValues[n][p] = float(val)
    # for Multiregion blocks
    if tipe == "left_bdy":
        pTypes[n][p] = "Left boundary cond."
        pValues[n][p] = val
    if tipe == "right_bdy":
        pTypes[n][p] = "Right boundary cond."
        pValues[n][p] = val
    if tipe == "dy":
        pTypes[n][p] = "Buckling height"
        pValues[n][p] = float(val)
    if tipe == "dz":
        pTypes[n][p] = "Buckling depth"
        pValues[n][p] = float(val)
    if tipe == "origin":
        pTypes[n][p] = "Left boundary origin"
        pValues[n][p] = float(val)
    if tipe == "cellmix":
        pTypes[n][p] = "Cell mixture"
        pValues[n][p] = val
    # for DoubleHet blocks
