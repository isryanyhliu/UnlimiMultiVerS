import os
import json

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置相对路径
# input_file = os.path.join(script_dir, '../data_train/target/healthver/corpus.jsonl')
# output_file = os.path.join(script_dir, '../data_train/target/healthver/corpus_new.jsonl')

input_file = os.path.join(script_dir, '../data/healthver/corpus.jsonl')
output_file = os.path.join(script_dir, '../data/healthver/corpus_new.jsonl')

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析每一行的 JSON 数据
        data = json.loads(line.strip())
        
        # 获取 abstract 列表
        abstract = data.get('abstract', [])
        
        # 复制 abstract 内容 4 次
        extended_abstract = abstract * 3
        
        # 更新数据中的 abstract 字段
        data['abstract'] = extended_abstract
        
        # 将处理后的数据写入输出文件
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Processing complete. The new data has been saved to {output_file}.")
