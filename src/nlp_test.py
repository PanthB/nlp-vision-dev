import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print(f"[{time.time():.3f}] Starting NLP-VisionRT Command Processor...")
start_time = time.time()

print(f"[{time.time():.3f}] Loading sentence transformer model...")
model_start = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
model_load_time = time.time() - model_start
print(f"[{time.time():.3f}] Model loaded in {model_load_time:.3f} seconds")

print(f"[{time.time():.3f}] Defining command database...")
# Command database - each command maps to specific FPGA operations
commands_db = {
    "grayscale_enable": [
        "enable grayscale",
        "turn on grayscale", 
        "make everything black and white",
        "turn it grey",
        "switch to monochrome",
        "convert to grayscale",
        "remove color information"
    ],
    "grayscale_disable": [
        "disable grayscale",
        "turn off grayscale",
        "make it colorful again", 
        "restore original colors",
        "switch back to color mode",
        "enable color",
        "bring back colors"
    ],
    "blur_enable": [
        "enable blur",
        "make it blurry",
        "add blur effect",
        "soften the image",
        "apply gaussian blur",
        "make it fuzzy"
    ],
    "brightness_increase": [
        "make it brighter",
        "increase brightness",
        "brighten the image",
        "more light",
        "lighter image"
    ],
    "brightness_decrease": [
        "make it darker",
        "decrease brightness", 
        "darken the image",
        "less light",
        "darker image"
    ]
}

print(f"[{time.time():.3f}] Pre-computing embeddings for command database...")
encode_start = time.time()

# Pre-compute embeddings for all commands
command_embeddings = {}
for command_type, phrases in commands_db.items():
    print(f"[{time.time():.3f}]   Encoding {len(phrases)} phrases for '{command_type}'...")
    command_embeddings[command_type] = model.encode(phrases)

total_encode_time = time.time() - encode_start
print(f"[{time.time():.3f}] All embeddings computed in {total_encode_time:.3f} seconds")

FPGA_REGISTERS = {
    "grayscale_enable": {"address": 0x43C00000, "value": 1},
    "grayscale_disable": {"address": 0x43C00000, "value": 0}, 
    "blur_enable": {"address": 0x43C00004, "value": 2.5},  # sigma value
    "brightness_increase": {"address": 0x43C00008, "value": 25},  # +25 brightness
    "brightness_decrease": {"address": 0x43C00008, "value": -25}  # -25 brightness
}

def process_user_input(user_text):
    """Process user input and return the best matching command"""
    print(f"[{time.time():.3f}] Processing: '{user_text}'")
    
    # Generate embedding for user input
    embed_start = time.time()
    user_embedding = model.encode([user_text])
    embed_time = time.time() - embed_start
    print(f"[{time.time():.3f}] User embedding generated in {embed_time:.6f} seconds")
    
    # Find best match across all command types
    best_command = None
    best_score = 0.0
    all_scores = {}
    
    print(f"[{time.time():.3f}] Computing similarities...")
    for command_type, embeddings in command_embeddings.items():
        sim_start = time.time()
        similarities = cosine_similarity(user_embedding, embeddings)
        max_similarity = similarities.max()
        sim_time = time.time() - sim_start
        
        all_scores[command_type] = max_similarity
        print(f"[{time.time():.3f}]   {command_type}: {max_similarity:.3f} (computed in {sim_time:.6f}s)")
        
        if max_similarity > best_score:
            best_score = max_similarity
            best_command = command_type
    
    return best_command, best_score, all_scores

def execute_command(command_type):
    """Simulate FPGA register write"""
    if command_type in FPGA_REGISTERS:
        reg_info = FPGA_REGISTERS[command_type]
        print(f"[{time.time():.3f}] → FPGA WRITE: Address 0x{reg_info['address']:08X} = {reg_info['value']}")
        # In real system: mmio.write(reg_info['address'], reg_info['value'])
    else:
        print(f"[{time.time():.3f}] → UNKNOWN COMMAND: {command_type}")

def main():
    setup_time = time.time() - start_time
    print(f"\n[{time.time():.3f}] === NLP-VisionRT Ready in {setup_time:.3f} seconds ===")
    print(f"[{time.time():.3f}] Commands: grayscale on/off, blur, brightness up/down")
    print(f"[{time.time():.3f}] Type 'quit' or 'exit' to stop\n")
    
    SIMILARITY_THRESHOLD = 0.6  # Threshold for valid commands
    
    while True:
        try:
            user_input = input(f"\n[{time.time():.3f}] Enter command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"[{time.time():.3f}] Shutting down NLP-VisionRT...")
                break
                
            if not user_input:
                continue
                
            process_start = time.time()
            best_command, best_score, all_scores = process_user_input(user_input)
            
            if best_score >= SIMILARITY_THRESHOLD:
                print(f"[{time.time():.3f}] ✓ MATCH: {best_command} (confidence: {best_score:.3f})")
                execute_command(best_command)
            else:
                print(f"[{time.time():.3f}] ✗ NO MATCH: Best was {best_command} ({best_score:.3f}) < threshold ({SIMILARITY_THRESHOLD})")
            
            process_time = time.time() - process_start
            print(f"[{time.time():.3f}] Total processing time: {process_time:.3f} seconds")
            
        except KeyboardInterrupt:
            print(f"\n[{time.time():.3f}] Interrupted by user")
            break
        except Exception as e:
            print(f"[{time.time():.3f}] Error: {e}")

if __name__ == "__main__":
    main()
