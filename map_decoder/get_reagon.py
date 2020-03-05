import os, json, csv
import sklearn


labeled_f = './labeled_data.json'
entity_f = './named_entity.csv'
res_list = []
uncognized = []

with open(labeled_f) as f:
    labeled_str = ''
    for l in f.readlines():
        labeled_str += l
    o_json = json.loads(labeled_str)
    with open(entity_f) as e_f:
        e_csv = csv.reader(e_f)
        e_list = []
        for i, row in enumerate(e_csv):
            if i is not 0:
                e_list.append([row[0], row[1].lower().split(' '), row[2]])
                res_list.append([])
        for o in o_json:
            for p in o['points_list']:
                flag = False
                low = p['words'].lower()
                for e in e_list:
                    for w in e[1]:
                        if w in low and not flag:
                            flag = True
                            p['words'] = e[2]
                            p.update({'time': o['time']})
                            res_list[int(e[0])].append(p)
                if not flag:
                    uncognized.append(p)
print(res_list)
sum = 0
for i in res_list:
    print(len(i))
    sum += len(i)
print(sum)
