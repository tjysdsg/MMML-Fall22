import json

with open("./data/WebQA_test_data/submission_score_textonly.json", "r") as infile:
    test_results = json.load(infile)

# 0 - 3463 is image-based questions
# 3464 - x is text-based questions
'''
for idx, qid in enumerate(test_results.keys()):
    if qid == 'd5cd56c80dba11ecb1e81171463288e9':
        print(idx)
        import pdb; pdb.set_trace()
'''

for idx, qid in enumerate(test_results.keys()):
    # image based question
    if idx <= 3464:
        source_candidates = [cand for cand in test_results[qid]['sources'] if isinstance(cand[0], int)]
    else:
        source_candidates = [cand for cand in test_results[qid]['sources'] if isinstance(cand[0], str)]
    source_candidates = sorted(source_candidates, key=lambda x: x[1], reverse=True)

    res_sources = []
    if idx <= 3464:
        for cand in source_candidates:
            if cand[1] > 0.999 and len(res_sources) < 2:
                res_sources.append(cand[0])
        if len(res_sources) == 0:
            res_sources.append(source_candidates[0][0])
    else:
        res_sources = [cand[0] for cand in source_candidates[:2]]

    test_results[qid]['sources'] = res_sources

with open("./data/WebQA_test_data/submission_textonly3.json", "w") as outfile:
    json.dump(test_results, outfile)