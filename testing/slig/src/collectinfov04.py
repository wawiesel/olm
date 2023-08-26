#
# support tools for automated ORIGEN library generation slig
#
#   by BRIAN J. ADE and BENJAMIN R. BETZLER
#      on 20 May 2014
#
# collect library information into a single tex document
#

import glob, datetime
import os, fnmatch, sys, re
import argparse
from inputUtils import *


class documentation:
    def __init__(
        self,
        path="./",
        latexecho="no",
        baseName="basedoc.v04.tex",
        libInfoFileName="libraryInformation.tex",
    ):
        self.path = path
        self.latexecho = latexecho

        # standard LaTeX entry for a library using \section to add TOC, different "description" format
        self.docentry = """\\clearpage
        \\noindent
        \\section*{{{0}}}\\
        \\addcontentsline{{toc}}{{subsection}}{{{0}}}
        \\begin{{addmargin}}[2em]{{2em}}
        \\begin{{description}}
        {1}
        \\end{{description}}
        \\end{{addmargin}}"""

        # LaTeX item for a bulleted list - staring on next lind
        self.itementry = "\\item[{0}:] \\hfill \\\\ {1}"

        # LaTeX entry for a figure
        self.figentry = """\\begin{{figure}}[htb]
          \centering
          \includegraphics[width=0.75\\textwidth,clip=true,trim=0.5in 0.5in 0.5in 3.35in]{{{0}}}
          \caption{{{2}}}
          \label{{fi:{1}}}
        \end{{figure}}"""

        # BRB - LaTeX entries for a table split into top, entries, bottom
        self.tableHeader = """{0}, SCALE/TRITON lattice cell calculation parameters in \
          Table \\ref{{tab:{1}}}
          \\begin{{table}}[h!]
          \\centering
          \\caption{{SCALE/TRITON lattice cell calculation information.\\label{{tab:{1}}}}}
          \\begin{{tabular}}{{l@{{\\hskip 1cm}}l@{{\\hskip 1cm}}l}}
          \\toprule
            Description         & Parameter [radii in cm] & Material \\\\
          \\midrule
            Total calculations & {2} & {3} \\\\"""

        self.tableEntry = """
            {0} & {1} & {2} \\\\"""

        self.tableFooter = """
          \\bottomrule
          \\end{tabular}
        \\end{table}"""
        # BRB - LaTeX entries for a table split into top, entries, bottom

        self.docentries = []
        self.counter = 0

        # values for base LaTeX file currently hardcoded, as well as the final .tex file name.
        # because .tex name is hardcoded, final PDF file is not a changeable option at this time.
        self.basedoc = open(baseName).read()
        self.newdocname = libInfoFileName

        self.latexspecialchars = {
            "%": "\%",
            "_": "\_",
            "->": "$\\rightarrow$",
            "#": "\#",
            "&": "\&",
        }

        # find the libinfo file and any graphics files
        self.ext = ".pdf"
        pattern = "*" + self.ext
        self.graphicsFiles = []
        for root, dirs, files in os.walk(self.path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    self.graphicsFiles.append(os.path.join(root, name))
        self.graphicsNames = [
            i.split(self.ext)[0].split("/")[-1] for i in self.graphicsFiles
        ]

        self.graphicsfolder = "graphics"
        os.system("mkdir ./{0}".format(self.graphicsfolder))
        for file in self.graphicsFiles:
            os.system("cp {0} ./{1}".format(file, self.graphicsfolder))

    def appendData(self, information, libname, inputFile):
        f = information
        f = f.lstrip("\n")
        # replace special characters
        refName, texLibName = libname, libname
        for char in self.latexspecialchars:
            f = f.replace("' ", "'")
            f = f.replace(char, self.latexspecialchars[char])
            texLibName = texLibName.replace(char, self.latexspecialchars[char])
            refName = refName.replace(char, "")
        f = f.replace("'[", "[")
        f = f.replace("\n'", " ")
        # replace the previously replaced "_" chars between {}
        f = re.sub(r"\{(.+?)\\_(.+?)\}", r"{\1_\2}", f)
        f = re.sub(r"\{(.+?)\\_(.+?)\}", r"{\1_\2}", f)
        f = re.sub(r"\{(.+?)\\_(.+?)\}", r"{\1_\2}", f)
        splitfile = f.split("[")
        libitems = []
        if splitfile[0] == "":
            splitfile = splitfile[1:]
        for item in splitfile:
            infoname, parts = item.split("]")[0], [
                " ".join(part.split()) for part in item.split("]")[1].split("|")
            ]
            libitems.append("  " + self.itementry.format(infoname, "\\\\".join(parts)))
        tempentry = self.docentry.format(texLibName, "\n".join(libitems))

        if "figure{" in tempentry:
            figsplit = re.split("(figure{)", tempentry)
            for j in range(0, len(figsplit)):
                if figsplit[j] == "figure{":
                    figparts = figsplit[j + 1].split("}", 1)[0].split(":")
                    thisfig = self.figentry.format(
                        figparts[0], figparts[0].split(".")[0], figparts[1]
                    )
                    figsplit[j + 1] = thisfig + figsplit[j + 1].split("}", 1)[1]
            tempentry = "".join(figsplit).replace("figure{", "")
        self.docentries.append(tempentry)

        # BRB - add the table to the LaTeX .tex file
        if tempentry.find("roprietary") == -1:
            pTs, pVs, pCs, aCs = extractScaleInput(inputFile)
            if pTs != True:  # else there is no information extracted
                # find and store the end of the 'Model Info' line
                tabSplit = self.docentries[self.counter].split("[Model Info:]")
                tabSplit = tabSplit[1].split("[Sources:]")
                tabSplit = tabSplit[0].split("\n")
                oldInfo = tabSplit[0]
                # build the new information (table following the reference to it)
                newInfo = self.tableHeader.format(
                    oldInfo, pVs[0][0], len(pVs) + aCs, "-"
                )
                for j in range(len(pVs)):  # for each lattice cell calculation
                    newInfo += "[0.2cm]"
                    if (
                        (j > 0)
                        & (pVs[j - 1][1 : pTs[j][0]] == pVs[j][1 : pTs[j][0]])
                        & (pCs[j - 1][1 : pTs[j][0]] == pCs[j][1 : pTs[j][0]])
                    ):
                        newInfo += self.tableEntry.format(
                            pTs[j][1], "similar to type " + str(j), "-"
                        )
                    else:
                        for k in range(pTs[j][0]):  # walk through the parameter list
                            newInfo += self.tableEntry.format(
                                pTs[j][k + 1], pVs[j][k + 1], pCs[j][k + 1]
                            )
                newInfo += self.tableFooter
                # add this new information to the file
                self.docentries[self.counter] = self.docentries[self.counter].replace(
                    oldInfo, newInfo
                )
        # BRB - add the table to the LaTeX .tex file
        self.counter = self.counter + 1

    def update(self):
        newdoc = self.basedoc.replace(
            "%<libinfo>", "\n\n".join(self.docentries) + "\n\n%<libinfo>"
        )
        newdoc = newdoc.replace("<date>", datetime.datetime.now().strftime("%b %d, %Y"))
        newdoc = newdoc.replace("<graphicsfolder>", self.graphicsfolder)

        print("\n Generating {0}.".format(self.newdocname))
        open(self.newdocname, "w").write(newdoc)
        if self.latexecho == None or self.latexecho == "no":
            os.system(
                "pdflatex -interaction=batchmode {0} 1>/dev/null".format(
                    self.newdocname
                )
            )
            os.system(
                "pdflatex -interaction=batchmode {0} 1>/dev/null".format(
                    self.newdocname
                )
            )
        elif self.latexecho == "yes":
            os.system("pdflatex {0}".format(self.newdocname))
            os.system("pdflatex {0}".format(self.newdocname))
        else:
            sys.exit("invalid option for latexecho, cannot continue")
        os.system("rm -r {0}".format(self.graphicsfolder))
