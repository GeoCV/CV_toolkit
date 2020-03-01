import os, json
import sklearn


reagon_f = './location.json'
res_list = []

with open(reagon_f) as f:
    reagon_json = f.readline()
    o_json = json.loads(reagon_json)
    for o in o_json:
        for p in o['points_list']:
            low = p['words'].lower()
            if 'reagan' in low or 'ronald' in low:
                p.update({'time': o['time']})
                res_list.append(p)
print(res_list)
