import argparse,ast,csv,hashlib,importlib.util,json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current',root/'run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
parser=argparse.ArgumentParser(description='Compare food preprocessing with official Inverse Cooking rules.')
parser.add_argument('--source', type=Path, default=root/'work/reference_build_vocab.py')
args=parser.parse_args()
source=args.source.read_text()
expected_sha='e5effafb6664961e0f3a1f4e66be5947eb8054e0883e3f371c44b7cf85344361'
if hashlib.sha256(source.encode()).hexdigest() != expected_sha:
    raise ValueError('Upstream source differs from the audited snapshot; review before comparing.')
tree=ast.parse(source)
ns={}
names={'get_ingredient','get_instruction','cluster_ingredients','remove_plurals'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'original_helpers','exec'),ns)
build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_vocab_recipe1m')
for n in ast.walk(build):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base_words' for t in n.targets):
        base_words=ast.literal_eval(n.value)
ri={'and':['&',"'n"], '':['%',',','.','#','[',']','!','?']}
rs={'and':['&',"'n"], '':['#','[',']']}
def parse(layer,detection):
    raw=[ns['get_ingredient'](d,ri) for d,v in zip(detection['ingredients'],detection['valid']) if len(d)>0 and v]
    instructions=[t for i in layer['instructions'] if (t:=ns['get_instruction'](i['text'],rs))]
    assert raw==m.valid_raw_ingredients(detection)
    assert instructions==m.valid_instructions(layer)
    return raw,instructions
paths=(root/'data/recipe1m/recipe1M_layers.tar.gz',root/'data/recipe1m/det_ingrs.json')
counts=Counter()
for layer,detection in m.paired_records(*paths):
    raw,ins=parse(layer,detection)
    if len(raw)<2 or len(ins)<2 or len(ins)>=20 or len(raw)>=20 or sum(map(len,ins))<20: continue
    if layer['partition']=='train': counts.update(raw)
for word in base_words:
    if word not in counts: counts[word]=1
counts,clusters=ns['cluster_ingredients'](counts)
counts,clusters=ns['remove_plurals'](counts,clusters)
mapping={a:k for k,n in counts.items() if n>=10 for a in clusters[k]}
print('Original vocabulary reconstructed',len(mapping),flush=True)
compositions=Counter(); added=Counter(); total=0
for layer,detection in m.paired_records(*paths):
    total+=1
    raw,ins=parse(layer,detection)
    labels=tuple(sorted({mapping[x] for x in raw if x in mapping}))
    if len(labels)<2 or len(ins)<2 or len(ins)>=20 or len(labels)>=20 or sum(map(len,ins))<20: continue
    compositions[labels]+=1
    if not (2<=len(raw)<20): added[labels]+=1
print(json.dumps({'source':total,'eligible':sum(compositions.values()),'unique':len(compositions),'newly_retained':sum(added.values())}),flush=True)
with (root/'work/food/canonical_ingredient_mapping.csv').open(encoding='utf-8-sig') as f:
    actual_mapping={r['alias']:r['canonical_ingredient'] for r in csv.DictReader(f)}
assert mapping==actual_mapping,'Vocabulary mismatch'
with (root/'work/food/unique_compositions.csv').open(encoding='utf-8-sig') as f:
    actual={tuple(r['ingredients'].split('|')):int(r['weight']) for r in csv.DictReader(f)}
assert compositions==actual,'Composition/weight mismatch'
report={'original_source_url':'https://github.com/facebookresearch/inversecooking/blob/master/src/build_vocab.py','original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_records':total,'eligible_recipes':sum(compositions.values()),'unique_compositions':len(compositions),'newly_retained_recipes':sum(added.values()),'all_record_text_cleaning_equal':True,'vocabulary_mapping_equal':True,'all_compositions_and_weights_equal':True}
(root/'work/original_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: all records, vocabulary aliases, final compositions and weights match the original rules.',flush=True)
