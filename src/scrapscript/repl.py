import http.server
import os
import urllib.parse

from scrapscript.compiler import *
from scrapscript.stdlib import deserialize, STDLIB, bencode

ASSET_DIR = os.path.dirname(__file__)


class ScrapReplServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        logger.debug("GET %s", self.path)
        parsed_path = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        logging.debug("PATH %s", parsed_path)
        logging.debug("QUERY %s", query)
        if parsed_path.path == "/repl":
            return self.do_repl()
        if parsed_path.path == "/eval":
            try:
                return self.do_eval(query)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
                return
        if parsed_path.path == "/style.css":
            self.send_response(200)
            self.send_header("Content-type", "text/css")
            self.end_headers()
            with open(os.path.join(ASSET_DIR, "style.css"), "rb") as f:
                self.wfile.write(f.read())
            return
        return self.do_404()

    def do_repl(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        with open(os.path.join(ASSET_DIR, "repl.html"), "rb") as f:
            self.wfile.write(f.read())
        return

    def do_404(self) -> None:
        self.send_response(404)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"""try hitting <a href="/repl">/repl</a>""")
        return

    def do_eval(self, query: Dict[str, Any]) -> None:
        exp = query.get("exp")
        if exp is None:
            raise TypeError("Need expression to evaluate")
        if len(exp) != 1:
            raise TypeError("Need exactly one expression to evaluate")
        exp = exp[0]
        tokens = tokenize(exp)
        ast = parse(tokens)
        env = query.get("env")
        if env is None:
            env = STDLIB
        else:
            if len(env) != 1:
                raise TypeError("Need exactly one env")
            env_object = deserialize(env[0])
            assert isinstance(env_object, EnvObject)
            env = env_object.env
        logging.debug("env is %s", env)
        monad = ScrapMonad(env)
        result, next_monad = monad.bind(ast)
        serialized = EnvObject(next_monad.env).serialize()
        encoded = bencode(serialized)
        response = {"env": encoded.decode("utf-8"), "result": str(result)}
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
        return

