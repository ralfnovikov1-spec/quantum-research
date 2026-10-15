"""Execute Qmem4 cells S3-S5 offline: FakeKingston + Stim instead of IBM."""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "daily"),
                os.path.join(HERE, "..", "03_code_hhmem")]
import fake_ibm
nb = json.load(open(os.path.join(HERE, "Qmem4.ipynb"), encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"]]
g = {"service": fake_ibm.FakeService(), "SamplerV2": fake_ibm.FakeSampler,
     "FOLDER": tempfile.mkdtemp(), "json": json, "glob": __import__("glob"),
     "hashlib": __import__("hashlib"), "datetime": __import__("datetime")}
exec(src[3], g)                                   # S3 budget + plan
g["QUBITS"] = g["QUBITS"][:30]                    # keep the offline run short
g["CIRCS"], g["INDEX"] = g["N"].build(g["hw"], g["QUBITS"])
exec(src[4].replace("SUBMIT = False", "SUBMIT = True"), g)   # S4 submit
exec(src[5], g)                                   # S5 analyse
print("OFFLINE NOTEBOOK RUN OK")
