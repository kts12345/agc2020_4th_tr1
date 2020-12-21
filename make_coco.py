import os
import json
import shutil

import imagesize
from pathlib import Path
from tqdm.auto import tqdm

def get_empty_data_():
    return  dict(
        info=dict(description=None, url=None, version=None, 
                  year=2020, contributor=None, date_created=None),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[], 
        annotations=[], 
        categories=[dict(id=0, name='swoon'), dict(id=1, name='dummy') ],
    )


def to_json_format(img_base_dir, img_list): 

    out_json_path = '/aichallenge/temp_dir/4th_anno.json'
    
    # 메타 정보를 생성한다
    data = get_empty_data_()
    class_name_to_id = {v['name']:v['id'] for v in data['categories']}
    
    # 이미지 정보를 추가한다
    for img_id, img_path in enumerate(tqdm(img_list, desc='to_coco_format')):
        w, h = imagesize.get(img_path)
        l = dict(id=img_id, 
                 file_name=os.path.relpath(img_path, img_base_dir), 
                 width=w, height=h, iscrow=0)
        data['images'].append(l) 

    # 저장
    with open(out_json_path, 'w') as f:
        json.dump(data, f)
        
    # 이미지 root aliasing
    ln_img_prefix = '/aichallenge/temp_dir/4th_dataset'
    
    # 혹시 있을지도 모를 이전 데이터 삭제
    try:
        os.unlink(ln_img_prefix)
    except:
        pass
    shutil.rmtree(ln_img_prefix, ignore_errors=True)
    
    # aliasing
    cmd = f'ln -s {img_base_dir} {ln_img_prefix}'
    os.system(cmd)
    
    return data, out_json_path, ln_img_prefix
        
    
    