import json
import base64
import os

def encode_files(file_paths):
    files_object = {}
    
    for path in file_paths:
        with open(path, 'rb') as file:
            content = file.read()
            base64_content = base64.b64encode(content).decode('utf-8')
            filename = os.path.basename(path)
            files_object[filename] = {
                "original_name": filename,
                "content": base64_content
            }
    
    return json.dumps(files_object, indent=2)



def decode_files(encoded_content, output_path)->list:
    decoded = json.loads(encoded_content)
    
    saved_files = []

    for filename, file_info in decoded.items():
        with open(os.path.join(output_path,filename), 'wb') as output_file:
            output_file.write(base64.b64decode(file_info['content']))
            saved_files.append(os.path.join(output_path,filename))

    return saved_files
