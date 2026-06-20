# ============================================================
# test_services.py  -  smoke-test the running SOA services
# Start the services first (python run_services.py), then:
#   python test_services.py
# ============================================================
import requests
import config


def url(name):
    return f"http://{config.SERVICE_HOST}:{config.SERVICES[name]}"


def main():
    print("1) Gateway health (checks every service):")
    print("  ", requests.get(url("gateway") + "/health").json())

    print("\n2) Preprocessing service:")
    print("  ", requests.post(url("preprocessing") + "/preprocess",
                              json={"text": "How to INVEST in stocks?"}).json())

    print("\n3) Indexing service stats:")
    print("  ", requests.get(url("indexing") + "/stats").json())

    print("\n4) Refinement service (typo):")
    print("  ", requests.post(url("refinement") + "/suggest",
                              json={"query": "invset in stock markett"}).json())

    print("\n5) Retrieval service (BM25):")
    r = requests.post(url("retrieval") + "/search",
                      json={"query": "how to invest in indian stock market",
                            "model": "BM25", "top_k": 3}).json()
    print("   ms:", r["ms"])
    for x in r["results"]:
        print("   -", round(x["score"], 2), x["text"][:60])

    print("\n6) Full pipeline via Gateway (refine + retrieve):")
    r = requests.post(url("gateway") + "/search",
                      json={"query": "how to invset in indian stock markett",
                            "model": "BM25", "top_k": 3, "refine": True}).json()
    for x in r["results"]:
        print("   -", round(x["score"], 2), x["text"][:60])


if __name__ == "__main__":
    main()
