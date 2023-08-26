#!/usr/bin/env python
# for romulus - !/usr/bin/env /opt/Python-2.7.6/bin/python
#
# SCALE/ORIGEN Library Inputs Generator
#
#   by BENJAMIN R. BETZLER
#      on 19 Jun 2014
#
#   edited by BRIANA D. HISCOX
#      on October 1st 2020
#
#  Edit list:
#     1.Added ('s to print statements several times
#     2.Removed the version command from argparse since it is not supported in this python
#     3.added "int" to line 666
#
# This python script semi-automates the four-step process necessary to create
# ORIGEN libraries with TRITON/NEWT. See sligManual.pdf for details.
#
#   note: executes main() at end of file
#
import os, fnmatch, re, sys, argparse, shutil
from collectinfov04 import documentation


def main():
    #
    title = "SCALE/ORIGEN Library Inputs Generator"
    #
    # command line arguments
    parser = argparse.ArgumentParser(description=title)
    #           version     = ':: v 1.3 by B. R. Betzler')
    #
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="generate input files and shell scripts (steps 1-2)",
    )
    #
    parser.add_argument(
        "-f",
        "--finish",
        action="store_true",
        help="finish, i.e., add libraries to arpdata.txt (step 4)",
    )
    #
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./",
        help="path to the template files (default = './')",
    )
    #
    parser.add_argument(
        "-x",
        "--xsections",
        type=str,
        default="v7-252",
        help="cross section library to be used (default = 'v7-252')",
    )
    #
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default="_template.inp",
        help="template file extension (default = '_template.inp')",
    )
    #
    parser.add_argument(
        "-a", "--add", action="store_true", help="add input files to the current run"
    )
    #
    parser.add_argument(
        "-d",
        "--document",
        action="store_true",
        help="generate documentation for each libraries",
    )
    #
    parser.add_argument(
        "-s",
        "--submit",
        action="store_true",
        help="generate submit scripts for running on clusters",
    )
    #
    args = parser.parse_args()
    screen = messenger(title)
    #
    # directory and file names
    root = os.popen("pwd").read().replace("\n", "") + "/"
    runDirectory = "runSpace"
    workDirectory = "workSpace"
    bankDirectory = "newLibraries"
    arpList = "arpLibList.txt"
    pbsName = "run1proc.pbs"
    submitName = "submitSLIGjobs"
    arpFile = "addToArpData.txt"
    tagFile = "tagEmUp.sh"
    libInfoFile = "libraryInformation.tex"
    suffix = ".inp"
    libExt = ".f33cmb"
    BUListFlag = "BurnupList"
    BUFlag = "Initial Bootstrap Calculation"
    feedArpFile = '" >> ' + root + arpFile + "\n"
    feedTagFile = '" >> ' + root + tagFile + "\n"
    clearArpFile = 'echo -n "" > ' + root + arpFile + "\n"
    clearTagFile = 'echo -n "" > ' + root + tagFile + "\n"
    #
    # pbs options for submission to queue
    splitSubmitScript = False  # split up the submit script
    sleepTime = "0"  # seconds between pbs script submissions
    pbsOptions = {}
    pbsOptions["numnode"] = "1"
    pbsOptions["numproc"] = "1"
    pbsOptions["timeest"] = "100:00:00"
    pbsOptions["pathtoscale"] = "/scale/staging/6.2-rev19189/bin/scalerte"
    pbsOptions["username"] = os.popen("whoami").read().replace("\n", "")
    #
    # verify arguments
    if args.generate and args.finish:
        message = "Two options are selected (-g and -f). SLIG cannot do both!"
        screen.printError(message)
    #
    if not args.generate and not args.finish:
        message = "No actions are specified (-g or -f). Use -h for options."
        screen.printError(message)
    #
    if args.finish and not os.path.isfile(arpList):
        message = (
            "'"
            + arpList
            + "' file not found in current directory!  "
            + "Either SLIG -g was not run or the file was removed."
        )
        screen.printError(message)
    #
    if args.submit and not os.path.exists(pbsName):
        message = (
            "'"
            + pbsName
            + "' needed to create submit scripts!  "
            + "Move the template to the current directory."
        )
        screen.printError(message)
    #
    # generate scale input files
    if args.generate:
        #
        screen.printMessage("Running SLIG: Generate input files!")
        xsLibParam = "xsLib"
        xsLib = args.xsections
        ext = args.extension
        #
        # find template files and check for their existence
        searchPath = manageDirectory(args.path, screen, option="no make")
        tempFiles = searchPath.findFiles("*" + ext)
        nTemplates = len(tempFiles)
        screen.printMessage(str(nTemplates) + " '*" + ext + "' template files found.")
        if nTemplates == 0:
            message = (
                "No templates found!  Use the -p PATH option to "
                + "specify the location of the templates."
            )
            screen.printError(message)
        tempNames = [i.split(ext)[0].split("/")[-1] for i in tempFiles]
        #
        # sort alphabetically
        sort = sorted(range(len(tempNames)), key=lambda k: tempNames[k].lower())
        templateFiles = []
        templateNames = []
        for i in range(len(sort)):
            templateFiles.append(tempFiles[sort[i]])
            templateNames.append(tempNames[sort[i]])
        #
        # generate subdirectories and input files for each template perturbation
        message = "\n Or use slig.py -ag to add calculations to this set."
        if args.add:
            if not os.path.exists(runDirectory):
                message = (
                    "Directory "
                    + runDirectory
                    + " does not exist!  "
                    + "Either SLIG -g was not run or the directory was "
                    + "removed."
                )
                screen.printError(message)
            runSpace = manageDirectory(runDirectory, screen, option="no make")
            if os.path.exists(libInfoFile):
                screen.printMessage("Adding documentation to " + libInfoFile + ".")
                args.document = True
                docFile = documentation(baseName=libInfoFile)
            else:
                args.document = False
            if os.path.exists(submitName):
                screen.printMessage("Generating submit scripts.")
                args.submit = True
            else:
                args.submit = False
            submitScript = manageFile(submitName + "RS", screen, option="no read")
            arpData = manageFile(arpFile, screen)
            listFile = manageFile(arpList, screen)
            splitSubmitScript = True
        else:
            runSpace = manageDirectory(runDirectory, screen, message=message)
            if args.document:
                docFile = documentation()
            submitScript = manageFile(submitName, screen, option="not read")
            arpData = manageFile(arpFile, screen, option="not read")
            listFile = manageFile(arpList, screen, option="not read")
            ###submitScript.appendLines(clearTagFile)
        workSpace = manageDirectory(workDirectory, screen)
        nRuns = 0
        screen.printMessage("Building directory tree and input files.")
        for i in range(nTemplates):
            #
            templateDirectory, exists = runSpace.makeDirectory(templateNames[i])
            if exists:
                absoluteLocation = runSpace.location + templateDirectory
                message = "Directory '" + absoluteLocation + "' exists."
                message += "\n Will not repeat calculations for this template."
                screen.printMessage(message)
                continue
            template = manageTemplate(templateFiles[i], screen)
            pbsScriptTemplate = manageTemplate(pbsName, screen)
            optionSets, optionIDs = template.getOptions(libExt)
            burnupLines, location = template.getBurnupSequence()
            if args.document:
                libInfo = template.extractDocumentation(optionSets, xsLibParam, xsLib)
                docFile.appendData(libInfo, templateNames[i], templateFiles[i])
            #
            for j in range(len(optionSets)):
                #
                # copy template to a new working file

                nRuns = nRuns + 1
                workFileName = workSpace.location + templateNames[i]
                template.duplicateTo(workFileName)
                workFile = manageFile(workFileName, screen)
                #
                # insert options into the working file
                newLines = workFile.fileLines
                newLines = [line.replace(xsLibParam, xsLib) for line in newLines]
                for option in optionSets[j]:
                    for k in range(len(newLines)):
                        if (newLines[k][0] != "'") and (option in newLines[k]):
                            newLines[k] = newLines[k].replace(
                                option, optionSets[j][option]
                            )
                workFile.fileLines = newLines
                workFile.insertLines(burnupLines, location[0])
                #
                # update and move the working file to a new run space directory
                newRunSpaceName, exists = runSpace.makeDirectory(
                    templateDirectory + optionIDs[j]
                )
                newInputRunSpace = runSpace.location + newRunSpaceName
                newInputName = newInputRunSpace + optionIDs[j] + suffix
                workFile.updateFile()
                workFile.moveFileTo(newInputName)
                pbsOptions["inputname"] = optionIDs[j] + suffix
                if optionIDs[j] == "only":
                    optionIDs[j] = ""
                #
                # copy pbs (submit) script template to workspace
                if args.submit:
                    workScriptName = workSpace.location + pbsName
                    pbsScriptTemplate.duplicateTo(workScriptName)
                    workFile = manageFile(workScriptName, screen)
                    #
                    # insert options into the working pbs script
                    for option in pbsOptions:
                        newLines = [
                            line.replace(option, pbsOptions[option])
                            for line in workFile.fileLines
                        ]
                        workFile.fileLines = newLines
                    #
                    # update and move the pbs script to the run space
                    newpbsName = newInputRunSpace + pbsName
                    workFile.updateFile()
                    workFile.moveFileTo(newpbsName)
                    #
                    # build a bash script to submit runs to the queue
                    submitScript.appendLines("cd " + root + newInputRunSpace + "\n")
                    submitScript.appendLines("qsub " + pbsName + "\n")
                listFile.appendLines(
                    str(newInputRunSpace)
                    + str(optionSets[j][template.libNameParam])
                    + "\n"
                )

            #
            # to build entries for the addition to the arpdata.txt file and
            #     a script for tagging arpdata.txt information to binaries
            echo = 'echo " %s' + feedArpFile
            echoN = 'echo -n " %s' + feedArpFile
            idTag = "-idtags='assembly=%s'" % templateNames[i]
            interpTag = "-interptags='%s=%s'"
            arpData.appendLines("!" + templateNames[i] + "\n")
            if template.moxTemplate:
                arpData.appendLines(" " + str(len(template.puContent)))
                arpData.appendLines(" " + str(len(template.puVector)))
                arpData.appendLines(" 1")
                arpData.appendLines(" " + str(len(template.moderatorList)))
                arpData.appendLines(" " + str(len(template.burnups)) + "\n")
                arpData.appendLines(" " + " ".join(template.puContent) + "\n")
                arpData.appendLines(" " + " ".join(template.puVector) + "\n")
                arpData.appendLines(" 1.0\n")
                tagList = [
                    ["puContent", x, "puVector", y, "modDensity", z]
                    for x in template.puContent
                    for y in template.puVector
                    for z in template.moderatorList
                ]
            else:
                arpData.appendLines(" " + str(len(template.enrichmentList)))
                arpData.appendLines(" " + str(len(template.moderatorList)))
                arpData.appendLines(" " + str(len(template.burnups)) + "\n")
                arpData.appendLines(" " + " ".join(template.enrichmentList) + "\n")
                tagList = [
                    ["enrichment", x, "modDensity", y]
                    for x in template.enrichmentList
                    for y in template.moderatorList
                ]
            arpData.appendLines(" " + " ".join(template.moderatorList) + "\n")
            # reorder optionIDs for MOX templates
            if template.moxTemplate:
                reorderedList = []
                for j in range(len(template.moderatorList)):
                    for k in range(len(template.puVector)):
                        for l in range(len(template.puContent)):
                            index = (
                                l * len(template.puVector) * len(template.moderatorList)
                                + k * len(template.moderatorList)
                                + j
                            )
                            reorderedList.append(optionIDs[index])
                optionIDs = reorderedList
            for j in range(len(optionIDs)):
                libName = "'" + template.libName + optionIDs[j] + libExt + "'"
                ###tag   = "echo \"obtagmod %s " + libName + feedTagFile
                ###submitScript.appendLines(tag % idTag)
                if j + 1 == len(optionIDs) or (j + 1) % 3 == 0:
                    arpData.appendLines(" " + libName + "\n")
                else:
                    arpData.appendLines(" " + libName)
                ###for k in range(len(tagList[0])/2):
                ###    newTag = interpTag % (tagList[j][2*k],tagList[j][2*k+1])
                ###    submitScript.appendLines(tag % newTag)
            arpData.appendLines(" " + str(templateNames[i] + BUListFlag) + "\n")
            submitScript.appendLines("sleep " + sleepTime + "\n")
            #
            # for smaller submit scripts
            if args.submit and splitSubmitScript:
                offset = 1
                while os.path.exists(root + submitName + str(offset)):
                    offset += 1
                submitScript.appendLines(
                    "chmod -x " + root + submitName + str(offset) + "\n"
                )
                submitScript.updateFile()
                submitScript.changeMode("+x")
                submitScript.moveFileTo(submitName + str(offset))
                submitScript = manageFile(submitName + "RS", screen, option="not read")
        #
        # write the submit script to file, clean up directory
        if args.submit and not splitSubmitScript:
            submitScript.appendLines("chmod -x " + root + submitName + "\n")
            submitScript.updateFile()
            submitScript.changeMode("+x")
        if args.document:
            docFile.update()
        arpData.updateFile()
        listFile.updateFile()
        workSpace.removeDirectory("")
        screen.printMessage(str(nRuns) + " runs to be submitted to queue.")
        screen.printMessage(
            "SLIG --generate complete. Run SCALE and/or " + submitName + ".\n"
        )
        #
        return
    #
    # collect libraries and form library list
    if args.finish:
        #
        screen.printMessage("Running SLIG: Finish up the run!")
        listFile = manageFile(arpList, screen)
        submitFile = manageFile(submitName, screen, option="not read")
        libraryList = [line.replace("\n", "") for line in listFile.fileLines]
        missingFiles = 0
        newLibraryList = []
        newOutputList = []
        #
        # check if SCALE has generated all libraries
        for line in libraryList:
            newLocation = "./" + bankDirectory + "/" + line.split("/")[-1]
            fileInRunSpace = os.path.isfile(line)
            fileInNewLocation = os.path.isfile(newLocation)
            if fileInRunSpace:
                newLibraryList.append(line)
                outName = line.split("/")[-2] + ".out"
                newOutputList.append(line.replace(line.split("/")[-1], outName))
            elif not fileInRunSpace and not fileInNewLocation:
                screen.printMessage("Missing library " + line)
                missingFiles += 1
        if missingFiles > 0:
            message = (
                str(missingFiles)
                + " missing libraries!  "
                + "Run SCALE to generate these libaries."
            )
            screen.printError(message)
        #
        # move libraries to a new directory
        nLibraries = len(newLibraryList)
        message = "Moving " + str(nLibraries) + " libraries to ./" + bankDirectory + "/"
        screen.printMessage(message)
        message = "\n Or use slig.py -af to add finished libraries to this set."
        if args.add:
            newLibraryDirectory = manageDirectory(
                bankDirectory, screen, option="no make"
            )
        else:
            newLibraryDirectory = manageDirectory(
                bankDirectory, screen, message=message
            )
        for line in newLibraryList:
            libFile = manageFile(line, screen)
            libFile.moveFileTo(newLibraryDirectory.location + line.split("/")[-1])
        #
        # read burnup list from outputs and write to arpdata.txt file
        screen.printMessage("Writing burnup lists to arpdata.txt file.")
        arpData = manageFile(arpFile, screen)
        burnups = {}
        for line in newOutputList:
            template = line.split("/")[-3]
            if template in burnups:
                continue
            tempBurnups = []
            outputFile = manageFile(line, screen)
            toss, indices = outputFile.getLines(BUFlag)
            i = 0
            while True:
                tempLine = outputFile.fileLines[indices[0] + i]
                if not tempLine.split()[0] == str(i):
                    break
                burnupValue = float(tempLine.split()[-1])
                tempBurnups.append(burnupValue)
                i += 1
            if template in burnups:
                if not burnups[template] == tempBurnups:
                    message = (
                        "Burnups for "
                        + template
                        + " are not equal for "
                        + "calculations.  Define finer burnup steps on "
                        + "on the template header file."
                    )
                    screen.printError(message)
            else:
                message = "Writing burnup list for " + template + "."
                screen.printMessage(message)
                burnups[template] = tempBurnups
        for template in burnups:
            burnupFlag = " " + template + BUListFlag
            burnupText = ""
            for i in range(len(burnups[template])):
                if i > 0 and i % 6 == 0:
                    burnupText += "\n " + str(burnups[template][i])
                else:
                    burnupText += " " + str(burnups[template][i])
            for i in range(len(arpData.fileLines)):
                newLine = arpData.fileLines[i].replace(burnupFlag, burnupText)
                arpData.fileLines[i] = newLine
        # screen.printMessage(burnupText)
        arpData.updateFile()
        screen.printMessage("SLIG --finish complete.\n")
        #
        return


#
# special class for specific work on SCALE-related templates
class manageTemplate:
    def __init__(self, name, screen):
        #
        self.template = manageFile(name, screen)
        self.fileName = name
        self.fileComments = []
        self.null = "null_string"
        self.screen = screen
        self.shortName = name.split("/")[-1]
        #
        # strip comments from file
        for line in self.template.fileLines:
            if line[0] == "'":
                self.fileComments.append(line)
        self.parameterList = self.extractInfo("param")
        self.sortParameters()

    def extractInfo(self, blockID):
        #
        infoList = {}
        insideBlock = False
        #
        # search comments for blockID
        for i in range(len(self.fileComments)):
            line = self.fileComments[i]
            if not line.startswith("'     "):
                insideBlock = False
            if line.find(blockID) != -1:
                insideBlock = True
            if insideBlock and " - " in line:  # extract values
                name = line.split(" - ")[0].split()[-1]
                values = line.split(" - ")[-1].replace(" ", "")
                j = 1
                while values.endswith(",\n"):  # for multiple-line inputs
                    values += self.fileComments[i + j]
                    values = values.replace(" ", "")
                    j = j + 1
                values = values.replace("\n", "").replace("'", "")
                infoList[name] = values.split(",")
        #
        return infoList

    def extractDocumentation(self, optionSets, libParam, xsLibrary):
        #
        documentation = ""
        insideBlock = False
        #
        # search comments for Documentation
        for i in range(len(self.fileComments)):
            line = self.fileComments[i]
            if line.startswith("' -----"):
                insideBlock = False
            if insideBlock:  # extract text
                documentation = documentation + line
            if line.find("Documentation") != -1:
                insideBlock = True
        #
        # populate empty fields with header information
        #
        if self.moxTemplate:
            tempList = ", ".join(self.puContent)
            contentInfo = " Pu Contents/Enrichment (%): " + tempList
            tempList = ", ".join(self.puVector)
            vectorInfo = "Pu Vector/Pu-239 Concentration (%): " + tempList
            dataRangeInfo = "[Data Range]" + contentInfo + "|" + vectorInfo
        else:
            tempList = ", ".join(self.enrichmentList)
            enrichInfo = " Enrichments: " + tempList
            dataRangeInfo = "[Data Range]" + enrichInfo
        tempList = ", ".join(self.moderatorList)
        densInfo = "Moderator densities (g/cc): " + tempList
        tempList = ", ".join(self.totBurnups)
        burnInfo = "Cumulative burnup steps (GWD/MTHM): " + tempList
        dataRangeInfo = dataRangeInfo + "|" + densInfo + "|" + burnInfo
        documentation = documentation.replace("[Data Range]", dataRangeInfo)
        powerInfo = "[Power] " + self.power + " MW/MTHM"
        documentation = documentation.replace("[Power]", powerInfo)
        tempRange = range(len(optionSets))
        tempList = [optionSets[i][self.libNameParam] for i in tempRange]
        libraryInfo = "[Libraries] " + str(len(optionSets)) + " files: | "
        libraryInfo = libraryInfo + ", ".join(tempList)
        documentation = documentation.replace("[Libraries]", libraryInfo)
        documentation = documentation.replace(libParam, xsLibrary)
        #
        return documentation

    def sortParameters(self):
        #
        self.moxTemplate, self.usingDancoff = False, False
        self.u234Used, self.u236Used = False, False
        self.u235Param, self.u238Param = self.null, self.null
        self.u234Param, self.u236Param = self.null, self.null
        self.puContentParam, self.puVectorParam = {}, self.null
        self.densityParam, self.am241Param = self.null, self.null
        self.dancoffParam, self.uContentParam = [], {}
        #
        for param in self.parameterList:
            if "235" in param or "235" in self.parameterList[param][0]:
                self.u235Param = param
            if "234" in param or "234" in self.parameterList[param][0]:
                self.u234Param = param
                self.u234Used = True
            if "236" in param or "236" in self.parameterList[param][0]:
                self.u236Param = param
                self.u236Used = True
            if "238" in param or "238" in self.parameterList[param][0]:
                if "pu" in param or "pu" in self.parameterList[param][0]:
                    self.pu238Param = param
                else:
                    self.u238Param = param
            if "239" in param or "239" in self.parameterList[param][0]:
                self.pu239Param = param
                self.moxTemplate = True
            if "240" in param or "240" in self.parameterList[param][0]:
                self.pu240Param = param
            if "241" in param or "241" in self.parameterList[param][0]:
                self.pu241Param = param
            if "242" in param or "242" in self.parameterList[param][0]:
                self.pu242Param = param
            if "dens" in param or "dens" in self.parameterList[param][0]:
                self.densityParam = param
            if "danc" in param or "danc" in self.parameterList[param][0]:
                self.dancoffParam.append(param)
                self.usingDancoff = True
            if "cont" in param or "cont" in self.parameterList[param][0]:
                if "plut" in param or "plut" in self.parameterList[param][0]:
                    zone = self.classifyZone(param, self.parameterList[param][0])
                    self.puContentParam[zone] = param
                if "uran" in param or "uran" in self.parameterList[param][0]:
                    zone = self.classifyZone(param, self.parameterList[param][0])
                    self.uContentParam[zone] = param
            if "name" in param or "name" in self.parameterList[param][0]:
                self.libNameParam = param
            if "powe" in param or "powe" in self.parameterList[param][0]:
                self.powerParam = param
            if "days" in param or "days" in self.parameterList[param][0]:
                self.dayParam = param
            if "Am" in param or "americium" in self.parameterList[param][0]:
                self.am241Param = param

    def getBurnupSequence(self):
        #
        # calculate length of each burnup step
        burnupList = [float(x) * 1e3 for x in self.burnups]
        daysToBurn = [burnup / float(self.power) for burnup in burnupList]
        if burnupList[0] > 0:
            message = "Burnup step for 0.0 GWd/MTHM added (%s)."
            self.screen.printMessage("Warning: " + message % self.shortName)
            burnupList.insert(0, 0.0)
            daysToBurn.insert(0, 0.0)  # first step of two days for Xe equilibrium
        daysPerStep = []
        if daysToBurn[1] > 3:
            message = "Burnup step added (%s).\n Initial step is > 3 days."
            self.screen.printMessage("Warning: " + message % self.shortName)
            daysToBurn.insert(1, 2)  # first step of two days for Xe equilibrium
            burnupList.insert(1, 2 * float(self.power))
        midpointBurn = ["0"]
        for i in range(len(daysToBurn) - 1):
            daysPerStep.append(str(daysToBurn[i + 1] - daysToBurn[i]))
            midpointBurn.append(str((burnupList[i + 1] + burnupList[i]) / 2))
        #
        # build the new burndata card
        line, lineRange = self.template.getLines("power=")
        line = line.replace(self.powerParam, self.power)
        newLine = ""
        for days in daysPerStep:
            newLine += line.replace(self.dayParam, days)
        #
        # reassign midpoint burnups
        self.totBurnups = self.burnups
        self.burnups = midpointBurn
        #
        return newLine, lineRange

    def getOptions(self, ext):
        #
        # sort options into appropriate categories
        templateOptions = self.extractInfo("options")
        enrID, vectorID, modID = [""], [""], [""]
        dancoffList, dancoffPairs = [], []
        if len(templateOptions) == 3:  # no enrichments or densities
            self.enrichmentList, self.moderatorList = ["N/A"], ["N/A"]
            self.libName = templateOptions[templateOptions.keys()[1]][0]
            self.power = templateOptions[templateOptions.keys()[2]][0]
            self.burnups = templateOptions[templateOptions.keys()[0]]
            optionSets = [{self.libNameParam: self.libName + ext}]
            return optionSets, ["only"]
        for option in templateOptions:  # assign options
            if option.startswith("enrich"):
                self.enrichmentList = templateOptions[option]
                materialList = self.getConcentrations(templateOptions[option])
                enrID = ["e" + x.replace(".", "")[0:2] for x in self.enrichmentList]
                if len(enrID) == 1:
                    modID = [""]
            if option.startswith("content"):
                self.puContent = templateOptions[option]
                self.puVector = templateOptions["vector"]
                if "pin_zone" in templateOptions:
                    self.pinsPerZone = templateOptions["pin_zone"]
                    self.gadPins = templateOptions["pin_gad"]
                    self.pinDensity = templateOptions["avg_pin_dens."]
                materialList = self.getConcentrations()
                enrID = ["e" + x.replace(".", "")[0:2] for x in self.puContent]
                puSets, self.contents = self.getMoxContents(self.puContent)
            if option.startswith("vector"):
                vectorID = ["v" + x.replace(".", "")[0:2] for x in self.puVector]
                if len(vectorID) == 1:
                    modID = [""]
            if option.startswith("dens"):
                self.moderatorList = templateOptions[option]
                modID = ["w" + x.replace(".", "")[0:2] for x in self.moderatorList]
                if len(modID) == 1:
                    modID = [""]
            if option.startswith("danc"):
                dancoffList.append(templateOptions[option])
            if option.startswith("name"):
                self.libName = templateOptions[option][0]
            if option.startswith("power"):
                self.power = templateOptions[option][0]
            if option.startswith("burn"):
                self.burnups = templateOptions[option]
        #       print('the burnup is {}'.format(self.burnups))
        #
        if self.usingDancoff:  # pair dancoff factors
            for i in range(len(dancoffList[0])):
                tempList = []
                for j in range(len(dancoffList)):
                    tempList.append(dancoffList[j][i])
                dancoffPairs.append(tempList)
        # for each perturbation, build a list of parameters and their value
        if self.moxTemplate:
            optionSets = [
                {
                    self.pu238Param: y[0],
                    self.pu239Param: y[1],
                    self.pu240Param: y[2],
                    self.pu241Param: y[3],
                    self.pu242Param: y[4],
                    self.am241Param: y[5],
                    self.densityParam: z,
                }
                for x in self.contents
                for y in materialList
                for z in self.moderatorList
            ]
            totalSets = len(optionSets)
            totalPuSets = len(puSets)
            for i in range(totalSets):
                j = int(i / (totalSets / totalPuSets))
                # print(' j is {}'.format(j))
                optionSets[i].update(puSets[j])
            dancoffSets = [
                z for x in self.contents for y in materialList for z in dancoffPairs
            ]
        else:
            optionSets = [
                {
                    self.u235Param: x[0],
                    self.u234Param: x[1],
                    self.u236Param: x[2],
                    self.u238Param: x[3],
                    self.densityParam: y,
                }
                for x in materialList
                for y in self.moderatorList
            ]
            dancoffSets = [y for x in materialList for y in dancoffPairs]
        #
        optionIDs = [x + y + z for x in enrID for y in vectorID for z in modID]
        for i in range(len(optionSets)):  # assign unique library names
            optionSets[i][self.libNameParam] = self.libName + optionIDs[i] + ext
        for i in range(len(optionSets)):  # assign dancoff factors
            for j in range(len(self.dancoffParam)):
                optionSets[i][self.dancoffParam[j]] = dancoffSets[i][j]
        #
        return optionSets, optionIDs

    def getConcentrations(self, enrichmentList=""):
        #
        materialList = []
        self.Apu = []
        if self.moxTemplate:
            for pu239 in self.puVector:
                #
                # second-order polynomial form for Pu concentrations in percent
                a238, b238, c238 = 0.0045678, -0.66370, 24.9410
                a240, b240, c240 = -0.0113290, 1.02710, 4.7929
                a241, b241, c241 = 0.0018630, -0.42787, 26.3550
                a242, b242, c242 = 0.0048985, -0.93553, 43.9110
                pu238 = a238 * float(pu239) ** 2 + b238 * float(pu239) + c238
                pu240 = a240 * float(pu239) ** 2 + b240 * float(pu239) + c240
                pu241 = a241 * float(pu239) ** 2 + b241 * float(pu239) + c241
                pu242 = a242 * float(pu239) ** 2 + b242 * float(pu239) + c242
                norm = (pu238 + pu240 + pu241 + pu242) / (100 - float(pu239))
                pu238, pu240 = str(pu238 / norm)[0:9], str(pu240 / norm)[0:9]
                pu241, pu242 = str(pu241 / norm)[0:9], str(pu242 / norm)[0:9]
                if float(pu239) == 50:
                    am241 = "0.1595"
                elif float(pu239) == 55:
                    am241 = "0.1405"
                elif float(pu239) == 60:
                    am241 = "0.1230"
                elif float(pu239) == 65:
                    am241 = "0.1071"
                elif float(pu239) == 70:
                    am241 = "0.0928"
                materialList.append([pu238, pu239, pu240, pu241, pu242, am241])
                self.Apu.append(
                    (
                        float(pu238) * 238.0
                        + float(pu239) * 239.0
                        + float(pu240) * 240.0
                        + float(pu241) * 241.0
                        + float(pu242) * 242.0
                    )
                    / 100.0
                )
            #
        else:  # determine uranium isotopics from natural concentrations
            for enrichment in enrichmentList:
                uraniumRemaining = 100.0 - float(enrichment)
                u236wt, u234wt = self.null, self.null
                if self.u236Used:
                    u234wt = 0.007731 * float(enrichment) ** 1.0837
                    u236wt = 0.0046 * float(enrichment)
                    u238wt = uraniumRemaining - u234wt - u236wt
                    u234wt = str(u234wt)[0:9]
                    u236wt = str(u236wt)[0:9]
                    u238wt = str(u238wt)[0:9]
                elif self.u234Used:
                    u234wt = 0.007731 * float(enrichment) ** 1.0837
                    u238wt = uraniumRemaining - u234wt
                    u234wt = str(u234wt)[0:9]
                    u238wt = str(u238wt)[0:9]
                else:
                    u238wt = str(uraniumRemaining)
                materialList.append([enrichment, u234wt, u236wt, u238wt])
        #
        return materialList

    def classifyZone(self, parameter, definition):
        #
        # for MOX templates
        if "inner" in parameter or "inner" in definition:
            zone = "inner"
        elif "insideedge" in parameter or "insideedge" in definition:
            zone = "insideedge"
        elif "edge" in parameter or "edge" in definition:
            zone = "edge"
        elif "corner" in parameter or "corner" in definition:
            zone = "corner"
        else:
            zone = "N/A"
        #
        return zone

    def getMoxContents(self, averageContent):
        #
        averageContent = [float(x) for x in averageContent]
        if "N/A" in self.puContentParam:
            puParam = self.puContentParam["N/A"]
            uParam = self.uContentParam["N/A"]
            contents = [[str(x / 100.0), str(1 - x / 100.0)] for x in averageContent]
            optionSets = [{puParam: x[0], uParam: x[1]} for x in contents]
        else:  # pu zoning calculation
            numberPins = [int(x) for x in self.pinsPerZone]
            gdPins = int(self.gadPins[0])
            pinDensity = float(self.pinDensity[0])
            if gdPins > 0:
                line, toss = self.template.getLines("gd2o3")
                gdPinTag = "uo2.*\\" + line.split()[1]
                lines, toss = self.template.getLines(gdPinTag)
                uPctGd = float(lines.split()[3])
                ratios = [1.0, 0.75, 0.5, 0.3]  # inner/inside edge/edge/corner
            else:
                uPctGd = 0.0
                ratios = [1.0, 0.9, 0.68, 0.5]  # inner/inside edge/edge/corner
            optionSets = []
            Au = 238.0289  # g/mol
            Ao = 15.999  # g/mol
            order = ["inner", "insideedge", "edge", "corner"]
            denom = 0
            for i in range(len(ratios)):
                denom += ratios[i] * numberPins[i]
            contents = [[str(x / 100.0), str(1 - x / 100.0)] for x in averageContent]
            for content in averageContent:
                avgApu = sum(self.Apu) / len(self.Apu)
                Ahm = ((100 - content) * Au + content * avgApu) / 100.0
                hmInOnePin = Ahm / (Ahm + 2 * Ao) * pinDensity
                hmInPins = hmInOnePin * (sum(numberPins) + uPctGd * gdPins)
                puInPins = hmInPins * content / 100.0
                innerPuContent = 100.0 / hmInOnePin * puInPins / denom
                puContents = [innerPuContent * x for x in ratios]
                uContents = [str(1 - x / 100.0) for x in puContents]
                puContents = [str(x / 100.0) for x in puContents]
                optionSet = {}
                for i in range(len(puContents)):
                    optionSet[self.puContentParam[order[i]]] = puContents[i]
                    optionSet[self.uContentParam[order[i]]] = uContents[i]
                optionSets.append(optionSet)
        #
        return optionSets, contents

    def duplicateTo(self, newName):
        #
        self.template.copyFileTo(newName)


#
# manage prints to the screen
class messenger:
    def __init__(self, title):
        #
        print("")
        print(" " + title)

    def printMessage(self, message):
        #
        print("")
        print(" " + message)

    def printError(self, message):
        #
        print("")
        print(" Error: " + message)
        self.endMessage()

    def endMessage(self):
        #
        print("")
        print(" Exited due to errors.")
        print("")
        try:
            exit()
        except:  # for backwards compatibility
            sys.exit()


#
# creates directory tree
class manageDirectory:
    def __init__(self, name, screen, option="make", message=""):
        #
        self.location = "./" + name + "/"
        if os.path.exists(name) and option == "make":
            printMessage = "Please remove/rename the './" + name + "' directory. "
            screen.printError(printMessage + message)
        elif option == "make":
            self.makeDirectory("")
        elif not os.path.exists(name):
            printMessage = "Directory './" + name + "' does not exist."
            screen.printError(printMessage + message)

    def makeDirectory(self, name):
        #
        newDirectoryName = self.location + name
        if os.path.exists(newDirectoryName):
            return name + "/", True
        try:
            os.system("mkdir {0}".format(newDirectoryName))
        except:  # for backwards compatibility
            os.system("mkdir %s%s" % (newDirectoryName))
        return name + "/", False

    def removeDirectory(self, name):
        #
        if os.path.exists(self.location + name):
            os.removedirs(self.location + name)

    def findFiles(self, pattern):
        #
        # return list of files in the subdirectories that fit pattern
        fileList = []
        for root, dirs, files in os.walk(self.location):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    fileList.append(os.path.join(root, name))
        return fileList


#
# reads, writes/edits, and searches files
class manageFile:
    def __init__(self, name, screen, header=None, option="read"):
        #
        self.fileName = name
        self.fileHeader = header
        self.option = option
        #
        # read in the file for editing if it exists
        if os.path.isfile(self.fileName) and self.option == "read":
            self.fileStream = open(self.fileName, "r")
            self.fileLines = self.fileStream.readlines()
            # check for DOS control characters
            if "\r\n" in self.fileLines[0]:
                self.fileLines = [x.replace("\r\n", "\n") for x in self.fileLines]
            self.fileStream.close()
        else:
            self.fileLines = []
            if header != None:
                self.appendLines(self.fileHeader)
        #
        # print self.fileLines
        self.deletedLinesIndices = []

    def getLines(self, marker):
        #
        if marker == "all":
            return self.fileLines, len(self.fileLines)
        else:  # return the line contains the marker
            for i in range(len(self.fileLines)):
                if re.search(marker, self.fileLines[i]):
                    line = self.fileLines[i]
                    #
                    # read subsequent continuation lines
                    j = i + 1
                    while 1 and j < len(self.fileLines):
                        if self.fileLines[j].startswith("     "):
                            line = line + "\n" + self.fileLines[j]
                            j = j + 1
                        else:
                            break
                    return line, [i, j]
            print("Err: line with '" + marker + "' not found in " + self.fileName)
            try:
                exit()
            except:  # for backwards compatibility
                sys.exit()

    def removeLines(self, number):
        #
        # add line number(s) to remove list
        #
        if isinstance(number, int):
            self.deletedLinesIndices.append(number)
        else:
            for i in range(number[0], number[1]):
                self.deletedLinesIndices.append(i)

    def insertLines(self, newLine, number):
        #
        self.fileLines[number] = newLine

    def appendLines(self, newLine):
        #
        if isinstance(newLine, str):
            self.fileLines.append(newLine)
        else:
            for line in newLine:
                self.fileLines.append(line)

    def updateFile(self):
        #
        # rewrite all lines to file
        #
        fileStream = open(self.fileName, "w")
        for i in range(len(self.fileLines)):
            #
            # excluding lines marked for removal
            if not i in self.deletedLinesIndices:
                fileStream.write(self.fileLines[i])
            else:
                fileStream.write("c Note: a line was deleted here...\n")
        fileStream.close()

    def deleteFile(self):
        # i ain't got time to bleed
        if os.path.isfile(self.fileName):
            os.remove(self.fileName)

    def moveFileTo(self, newName):
        #
        if os.path.isfile(self.fileName):
            shutil.move(self.fileName, newName)

    def copyFileTo(self, newName):
        #
        if os.path.isfile(self.fileName):
            shutil.copy(self.fileName, newName)

    def changeMode(self, newMode):
        #
        if isinstance(newMode, str):
            os.system("chmod " + newMode + " " + self.fileName)


main()
