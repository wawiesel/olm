from pygments.lexer import RegexLexer, bygroups, include, inherit, words
from pygments.style import Style
from pygments import token
from pygments.lexers import get_lexer_by_name  # refer LEXERS
from pygments.lexers._mapping import LEXERS
from pygments.lexers.python import PythonLexer


################################################################################


def setup(app):
    app.add_lexer("scale", ScaleLexer)


################################################################################
class ScaleLexer(RegexLexer):
    name = "Scale"
    aliases = ["scale"]
    filenames = ["*.inp"]

    tokens = {
        "root": [
            (r"^=.*\n", token.Name.Function),
            (r"^\'.*\n", token.Comment),
            (r"^\‘.*\n", token.Comment),
            (r"%.*\n", token.Comment),
            (r"[ ]{1,}", token.Text),
            (r"\b(?i)(end([ ]{1,}|\n)end)\s\w+", token.Name, "block"),
            (r"\b(?i)(read|end)\s\w+", token.Name, "block"),
            (r"\b(?i)(end\s*$)", token.Name, "block"),
            (r"\b(?i)(location|cylgeometry|gridgeometry)\s", token.Name, "block"),
            (r"\b(?i)(energybounds|timebounds)\s", token.Name, "block"),
            (r"\b(?i)(response|distribution)\s", token.Name, "block"),
            (r"\b(?i)(pointdetector|regiontally|meshtally)\s", token.Name, "block"),
            (r"\b(?i)(src|meshsourcesaver)\s", token.Name, "block"),
            (
                r"\b(?i)(importancemap|adjointsource|macromaterial)\s",
                token.Name,
                "block",
            ),
            (r"\b(?i)(fill)\s", token.Name, "block"),
            (r"\b[0-9]+\s", token.Number),
            (r"([-+]?\d*\.?\d+)(?:[eE]([-+]?\d+))?\s", token.Number),
            (r"\"(.+?)\"", token.String),
            (r"\'(.+?)\'", token.String),
            (r"\"(.+?)\"", token.String),
            (r"\‘(.+?)\‘", token.String),
            (r"\"(.+?)\"", token.String),
            (r"\"(.+?)\"", token.String),
            (r"\"(.+?)\"", token.String),
            (r"\!.*\n", token.Comment),
            (r"(\w+|\n| )", token.Text),
            (
                r"(=|\-|\+|\%|\,|\‘|\$|\{|\}|\(|\)|\[|\]|\–|\_|\.|\…|\*|\,|\;|\:|\<|\>|\?|\/|\\)",
                token.Text,
            ),
            (r"\s+", token.Text),
            (r".* ", token.Text),
        ],
        "block": [
            (r"(\n|[ ]{0,}\n)", token.Text, "#pop"),
            (r"[a-zA-Z]+\s", token.Name, "#pop"),
            (r"[0-9]+\s", token.Number, "#pop"),
            (r"\!.*\n", token.Comment, "#pop"),
            (r".*\n", token.Text, "#pop"),
        ],
    }
