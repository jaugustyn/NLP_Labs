import ned, knowledge_graph as kg

print('--- NED Apple ---')
for c in ned.disambiguate('Apple', 'Apple released a new iPhone yesterday', 'en'):
    print(f"  {c.get('qid')} {c.get('label')}: {c.get('score'):.3f}")

print('--- NED Paris/France ---')
for c in ned.disambiguate('Paris', 'Paris is the capital of France', 'en'):
    print(f"  {c.get('qid')} {c.get('label')}: {c.get('score'):.3f}")

print('--- NED Paris/Hilton ---')
for c in ned.disambiguate('Paris', 'Paris Hilton attended the gala in New York', 'en'):
    print(f"  {c.get('qid')} {c.get('label')}: {c.get('score'):.3f}")

print('--- KG normalization ---')
ents = [
    [{'text': 'Henry', 'label': 'PERSON'}, {'text': 'Canada', 'label': 'GPE'}, {'text': 'Maria Rodriguez', 'label': 'PERSON'}],
    [{'text': 'Henry', 'label': 'PERSON'}, {'text': 'The Maple Leaves', 'label': 'WORK_OF_ART'}, {'text': 'Lucy', 'label': 'PERSON'}, {'text': 'the University of Toronto', 'label': 'ORG'}],
    [{'text': "Henry's", 'label': 'PERSON'}, {'text': 'Lucy', 'label': 'PERSON'}],
    [{'text': 'Henry later', 'label': 'PERSON'}, {'text': 'Lucy', 'label': 'PERSON'}],
]
G = kg.build_graph(ents)
print('nodes:', sorted(G.nodes))
out = kg.plot_graph(G, 'Knowledge Graph (en)', filename='test_kg2.png')
print('plot:', out)
