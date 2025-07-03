import re

input_path = "third_party/tflite-micro/tensorflow/lite/micro/examples/micro_speech/simple_features/model.cc"
output_path = "models/kws_arduino_og.tflite"

with open(input_path, "r") as file:
    content = file.read()

# Extract the model data using regex
hex_values = re.findall(r'0x[0-9a-fA-F]+', content)

# Convert to bytes
binary_data = bytes(int(x, 16) for x in hex_values)

# Write to .tflite file
with open(output_path, "wb") as f:
    f.write(binary_data)

print(f"Written to {output_path} ({len(binary_data)} bytes)")