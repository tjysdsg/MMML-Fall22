import json

with open("./data/WebQA_test_data/submission_score.json", "r") as infile:
    test_results = json.load(infile)

for qid in test_results.keys():
    test_results[qid]['sources'] = sorted(test_results[qid]['sources'], key=lambda x: x[1], reverse=True)
    test_results[qid]['sources'] = [x[0] for x in test_results[qid]['sources'] if x[1] > 0.6]

with open("./data/WebQA_test_data/submission.json", "w") as outfile:
    json.dump(test_results, outfile)