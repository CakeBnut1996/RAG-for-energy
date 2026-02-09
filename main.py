import yaml
from retrieval_utils.retriever import retrieve_data, rank_datasets
from generation_utils.generator import StudentGenerator
from generation_utils.schema import Response


def load_system():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    active_emb = cfg['embeddings'][cfg["retrieval"]['active_embedding']]
    active_stu = cfg['llm'][cfg["generation"]['active_student']]
    active_db = cfg['db'][cfg["retrieval"]['active_db']]

    sys_config = {
        "DB_PATH": cfg['data']['db_path'],
        "COLLECTION_NAME": active_db['collection'],
        "EMBEDDING_MODEL": active_emb['model'],
        "NUM_DOCS": cfg['retrieval']['num_docs'],
        "CHUNKS_PER_DOC": cfg['retrieval']['chunks_per_doc']
    }

    student_agent = StudentGenerator(
        provider=active_stu['provider'],
        model_name=active_stu['model']
    )
    return sys_config, student_agent


def main():
    sys_cfg, student = load_system()

    query_text = input("\n🔍 Query: ").strip()

    if query_text:
        try:
            # 1. Retrieval & Ranking
            retrieved_data = retrieve_data(
                query=query_text,
                db_path=sys_cfg["DB_PATH"],
                collection_name=sys_cfg["COLLECTION_NAME"],
                model_name=sys_cfg["EMBEDDING_MODEL"],
                num_docs=sys_cfg["NUM_DOCS"],
                chunks_per_doc=sys_cfg["CHUNKS_PER_DOC"]
            )
            ranked_data = rank_datasets(retrieved_data)

            # 2. Generation
            answer_object = student.generate(
                query=query_text,
                context=str(ranked_data),
                schema=Response
            )

            # 3. Raw CLI Output
            print("\n" + "=" * 50)
            print("🧠 ANSWER SUMMARY")
            print("=" * 50)
            print(f"{answer_object.answer}")
            print(f"\n📂 TOP SOURCE: {getattr(answer_object, 'name_top', 'N/A')}")

            print("\n" + "=" * 50)
            print("📊 SUPPORTING DATASETS")
            print("=" * 50)

            datasets = getattr(answer_object, 'supporting_datasets', [])
            if not datasets:
                print("No supporting datasets found.")
            else:
                for ds in datasets:
                    print(f"\n[📂 {ds.name}]")
                    print(f"Summary: {ds.summary}")
                    print(f"Quote: \"{ds.quote}\"")
                    print("-" * 30)

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()