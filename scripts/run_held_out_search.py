import csv
import os
import time

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.feasibility import candidate_cost

REPORTS_DIR = 'data/training_reports'
MODEL_PATH = 'models/full_dataset_agenttesla_vs_qbot_lgbm.pkl'
RESULTS_PATH = 'data/held_out_search_results.csv'

# Exact 80 held-out md5s and true labels, verbatim from the verified
# training run output (agenttesla_vs_qbot framing, seed=0, 317/80 split).
TEST_MD5S = ['d471fccf43598912d87a61afe9b6b984', 'f1e51386c9314a29be31514ef07849ed', 'ebae84848ff3ccdd3d7ef65add150494', 'c19155ad317de52254bbeca5a3ef183a', 'de8e768f352d14e6ae4a3f55e0bc6ec0', 'e79283f88b5442dae526598efe33a988', '2a65c38ce6f23978eafff6a3c8399eae', '75dda638ab0325fc1bd36449c7a2d523', 'eb89fd5ab96c4e20ceb891c3abbb09b0', '7bdbed830e3c183061a4e32d1cb3ec8e', '266bea059e22c1f41f1e52f91085557f', 'b3dfe02fbda31663dae8244cc72b5aa4', 'b938ad05451164f9955b3708258c3f69', '84e92db72cd1acab6ae6fcc4bcaba580', '192d67b731c7d2370090756d8dcb4bbb', '1fe73fe4d37cae6a02262b5164f3def0', 'a9b63c434e205092b3373e35c051a04a', 'c23d679d5d2be5be83719e676b001339', '3273a7981f07285e1ef1931ce323200b', '5af87bdc03cef91c14a7a6e8bdb74300', '4de31388247a9e1fed4c8a6ff24a6ac9', 'c61789e23f6f124e59c658a89d802667', '42dbb5cb5c0b2cbd7463b59ef87462c6', '132a2ba14ac1a95289b2aca07fd927d3', '864644f89cfa1aadb202f91c1bd24c11', '5ba695393b0cc69303e0679c5f60e8b4', '8749faaa0cd99cc1c11849ac401736e2', 'b640b0931f7fc701a5010b93675a2dee', 'cee2ad9f3f42786f3bf31316170eedc4', '005c26a27f4968f03c71ed5b6232dbea', 'd1fe1af58a4415d8cf2077859c54c890', 'de387ff820d4b468c4771b0da457e6e5', '9c1fb71fed14a2b8f5460b82129e03a3', '2010e737f4435fd3f46c0055ba44a73c', '02bf0fc6d6fdc5aa692f136da966b62c', 'c385df6dad6414c5834268634718ec62', '1318e8e6e1137db07e22e2e16662d721', '63ed8d0a4214ce7c0583a831afcbd8c0', 'bd52543d0b6bf874430cf5ee0ebf5fd4', 'a5ebc49571041789245437d1bd7bc271', 'fc24cf21d4fdc97c1ab364ed71cd00db', 'c422d49b3d3d8f5264fbcb26bdf32c26', '471975d5f2d8a1cbf65ee5664ae66ae7', 'd66294c92b8dfba0d840eaf3f9ed802e', 'cecdc5af3b097e4ea67f0d3bc5e3148d', '83a3340793cec4cb51526c5f4d6711b9', 'a550f57c45188ec167a8def3895c4828', '8f65eef8bfa87e7a9304c914940d50cf', '23bcc0472125f1974400e7f51a248e7f', 'c542abe2dc7aea52a097dd592ee9d7b7', 'b063e6f64f2de7c7c0b12207955a3620', '86c87c2d829f64c049266b9e0f85f319', 'e719222bd4624f1c0a92117fe33e2612', '1788b7df54735dcc51d7740b6a41ae8a', '2333295750647efb58f2e5541887cef3', '812861ad5cbb91bfa01a6a15c2cef128', '3711b0b15f26a0b23ab8f21fce1e095d', 'cc806cb9157aaae436fb3eafa8b9be56', 'ec2f47d8a5396b2c46f1cfc362c0dc6b', 'bfba2c5107bcb6cc4fb960361e5d8f7a', '5e28d0aedcfb4e7e344c8da36176ed97', 'd0c3b00cab92c1f2e17af864cc469518', '21c8c9b4fe3f3a655171bc51cd34e345', 'f922c0fe104d4b1fe24cd8fa90ed64c4', '700041a2722dd3976ea7a58823616265', '313fafcc8601c3fa149bd4192cf8482c', 'b7f6999b178db93cd5f9c05ed330aee2', '4ce54eda7650ff0f8062189f089b162e', 'c8d90ccd689b61c6b5f4978f5282a74f', '90ed98c5150194fc3b55f335ac61b943', 'b2fb08ffc4e31dbd2672637a96699ab7', '7dabb60027b6a37d05bb47434d49a9a9', '9e8d137e98395ae53b5e5a6c1f76a6aa', 'b3c3d4e0a644dc193c6a8dfec5975bba', 'ba25579c07a6f7540778a520e6587c84', '6398cc89d35a31b14a7fa0e6886b8430', '0c66d5af0dd882d40ce05e8121fdb0b5', 'a1cf058fbe5fe9f1704557a3703c7e25', 'fcb72c41cd799fef5bdd6f2865fe37dd', '97e744357997dda1ad4fdae4e773aed6']
TEST_LABELS = [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]

assert len(TEST_MD5S) == 80 and len(TEST_LABELS) == 80

FIELDNAMES = ['md5', 'true_label', 'status', 'orig_prob', 'new_prob', 'candidate_cost', 'elapsed_seconds']


def load_done_md5s():
    if not os.path.exists(RESULTS_PATH):
        return set()
    with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
        return {row['md5'] for row in csv.DictReader(f)}


def append_result(row):
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    done = load_done_md5s()
    if done:
        print(f'Resuming: {len(done)}/80 already completed, skipping those.')

    harness = ClassifierHarness(backend='lgbm', model_path=MODEL_PATH)

    remaining = [(md5, label) for md5, label in zip(TEST_MD5S, TEST_LABELS) if md5 not in done]
    print(f'{len(remaining)} samples left to process.')
    print('')

    for i, (md5, true_label) in enumerate(remaining, 1):
        t0 = time.time()
        path = os.path.join(REPORTS_DIR, f'{md5}.json')
        events = parse_cape_json(path)
        G = build_behavior_graph(events)

        search = CounterfactualSearch(classifier=harness, graph=G)
        result = search.run()

        elapsed = time.time() - t0
        status = result.get('status')
        orig_prob = result.get('orig_prob')
        new_prob = result.get('new_prob')
        cand = result.get('candidate')
        cost = candidate_cost(cand) if cand is not None else None

        row = {
            'md5': md5,
            'true_label': true_label,
            'status': status,
            'orig_prob': orig_prob,
            'new_prob': new_prob,
            'candidate_cost': cost,
            'elapsed_seconds': round(elapsed, 1),
        }
        append_result(row)

        print(f'[{i}/{len(remaining)}] {md5} (true={true_label}) -- status={status}, orig_prob={orig_prob}, new_prob={new_prob}, cost={cost}, {elapsed:.1f}s')

    print('')
    print('DONE. Results in', RESULTS_PATH)