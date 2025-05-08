import json

def write_manifest(output_filepath, final_list):
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        for entry in final_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
def read_json_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())  # parse each line as a JSON object
            data.append(record)
            
    return data
