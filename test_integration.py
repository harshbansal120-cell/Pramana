"""API integration tests run against TestClient (no live server needed)."""
import os
import sys
import tempfile
import pathlib

# Point data dir at a fresh temp dir so we don't pollute the project's data/
tmp = pathlib.Path(tempfile.mkdtemp())
os.environ["PRAMANA_DATA"] = str(tmp)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "backend"))

from fastapi.testclient import TestClient
from pramana.api import app

client = TestClient(app, raise_server_exceptions=True)
PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}{(' -- '+detail) if detail else ''}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}{(' -- '+detail) if detail else ''}")


print("\n=== Pramana API Integration Tests ===\n")

# 1. Health check
r = client.get("/healthz")
check("GET /healthz -> 200", r.status_code == 200)
check("GET /healthz body ok", r.json() == {"status": "ok"})

# 2. Status
r = client.get("/api/status")
check("GET /api/status -> 200", r.status_code == 200)
s = r.json()
check("Status has models key", "models" in s and "generator" in s["models"])
check("Status has kb key", "kb" in s and "sources" in s["kb"])
print(f"     KB: sources={s['kb']['sources']}, units={s['kb']['units']}, concepts={s['kb']['concepts']}")

# 3. Sources (empty)
r = client.get("/api/sources")
check("GET /api/sources -> 200 empty db", r.status_code == 200 and r.json() == [])

# 4. Search (empty db)
r = client.get("/api/search?q=probability")
check("GET /api/search -> 200 empty db", r.status_code == 200 and r.json() == [])

# 5. Students list (empty)
r = client.get("/api/students")
check("GET /api/students -> 200 empty", r.status_code == 200 and r.json() == [])

# 6. Create student (DB write)
r = client.post("/api/students", json={"name": "TestUser", "level": "beginner", "language": "en"})
check("POST /api/students -> 200", r.status_code == 200)
if r.status_code == 200:
    sid = r.json()["id"]
    check("Student has id", bool(sid))
    check("Student name correct", r.json()["name"] == "TestUser")

    # 7. Read back (persistence check)
    r2 = client.get("/api/students")
    check("Student persisted in DB", len(r2.json()) == 1)

    # 8. PATCH student
    r3 = client.patch(f"/api/students/{sid}", json={"name": "Renamed", "level": "intermediate", "language": "en"})
    check("PATCH /api/students -> 200", r3.status_code == 200)
    check("Name updated correctly", r3.json()["name"] == "Renamed")

# 9. SPA fallback
dist_index = pathlib.Path(__file__).parent / "frontend" / "dist" / "index.html"
if dist_index.exists():
    r = client.get("/")
    check("GET / -> SPA index.html 200", r.status_code == 200)
    check("Root content-type html", "text/html" in r.headers.get("content-type", ""))

    r = client.get("/library")
    check("GET /library -> SPA fallback 200", r.status_code == 200)
    check("SPA fallback returns html", "text/html" in r.headers.get("content-type", ""))

    r = client.get("/tutor")
    check("GET /tutor -> SPA fallback 200", r.status_code == 200)

    r = client.get("/practice")
    check("GET /practice -> SPA fallback 200", r.status_code == 200)
else:
    print("  SKIP  SPA tests -- build frontend first with: cd frontend && npm run build")

# 10. Eval endpoint (pre-computed results)
r = client.get("/api/eval")
check("GET /api/eval -> 200", r.status_code == 200)
check("Eval has pre-computed results", len(r.json()) > 0, f"keys={list(r.json().keys())}")

# 11. Unknown API route -> 404 not SPA
r = client.get("/api/not-a-real-endpoint")
check("Unknown /api/ route -> 404", r.status_code == 404)

# 12. Assessment with empty db -> 409 Conflict
r = client.post("/api/assessments", json={"student_id": "fake", "mode": "quiz", "n": 5})
check("Assessment empty db -> 409", r.status_code == 409, f"got {r.status_code}")

# 13. Graph endpoint (empty db)
r = client.get("/api/graph")
check("GET /api/graph -> 200 empty db", r.status_code == 200)
g = r.json()
check("Graph has nodes and edges keys", "nodes" in g and "edges" in g)

print(f"\n=== Results: {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    print("Failed tests:", FAIL)
    sys.exit(1)
else:
    print("All integration tests passed.")
