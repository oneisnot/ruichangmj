import time
import pickle
import multiprocessing

def encode_suit(counts):
    res = 0
    p = 1
    for c in counts:
        res += c * p
        p *= 5
    return res

def process_batch(states_batch):
    local_table = {}
    for base_counts in states_batch:
        counts = list(base_counts)
        max_t_no_pair = [-1] * 5
        max_t_pair = [-1] * 5
        
        def dfs(idx, g, t, has_pair):
            if g > 4: g = 4
            if t > 4 - g: t = 4 - g
            
            if has_pair:
                if t > max_t_pair[g]: max_t_pair[g] = t
            else:
                if t > max_t_no_pair[g]: max_t_no_pair[g] = t
                
            if idx >= 9:
                return

            if counts[idx] == 0:
                dfs(idx + 1, g, t, has_pair)
                return

            # Pair
            if counts[idx] >= 2 and not has_pair:
                counts[idx] -= 2
                dfs(idx, g, t, True)
                counts[idx] += 2
                
            # Triplet
            if counts[idx] >= 3:
                counts[idx] -= 3
                dfs(idx, g + 1, t, has_pair)
                counts[idx] += 3
                
            # Chow
            if idx <= 6 and counts[idx] > 0 and counts[idx+1] > 0 and counts[idx+2] > 0:
                counts[idx] -= 1; counts[idx+1] -= 1; counts[idx+2] -= 1
                dfs(idx, g + 1, t, has_pair)
                counts[idx] += 1; counts[idx+1] += 1; counts[idx+2] += 1
                
            # Pair as taatsu
            if counts[idx] >= 2:
                counts[idx] -= 2
                dfs(idx, g, t + 1, has_pair)
                counts[idx] += 2
                
            # Chow taatsu (n, n+1)
            if idx <= 7 and counts[idx] > 0 and counts[idx+1] > 0:
                counts[idx] -= 1; counts[idx+1] -= 1
                dfs(idx, g, t + 1, has_pair)
                counts[idx] += 1; counts[idx+1] += 1
                
            # Kanchan taatsu (n, n+2)
            if idx <= 6 and counts[idx] > 0 and counts[idx+2] > 0:
                counts[idx] -= 1; counts[idx+2] -= 1
                dfs(idx, g, t + 1, has_pair)
                counts[idx] += 1; counts[idx+2] += 1

            # Skip this tile kind
            temp = counts[idx]
            counts[idx] = 0
            dfs(idx + 1, g, t, has_pair)
            counts[idx] = temp

        dfs(0, 0, 0, False)
        
        encoded = encode_suit(base_counts)
        local_table[encoded] = (tuple(max_t_no_pair), tuple(max_t_pair))
        
    return local_table

def main():
    print("Finding all valid single-suit combinations (<=14 tiles)...")
    valid_states = []
    
    def get_valid_states(idx, current_counts, current_sum):
        if idx == 9:
            valid_states.append(list(current_counts))
            return
        for c in range(5):
            if current_sum + c <= 14:
                current_counts.append(c)
                get_valid_states(idx + 1, current_counts, current_sum + c)
                current_counts.pop()

    get_valid_states(0, [], 0)
    print(f"Total valid single-suit combinations: {len(valid_states)}")
    
    # Process all combinations using multiprocessing
    cpu_cores = multiprocessing.cpu_count()
    print(f"Generating Suit Table DP using {cpu_cores} cores...")
    start_time = time.time()
    
    chunk_size = len(valid_states) // cpu_cores + 1
    batches = [valid_states[i:i + chunk_size] for i in range(0, len(valid_states), chunk_size)]
    
    suit_table = {}
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        results = pool.map(process_batch, batches)
        for r in results:
            suit_table.update(r)

    print(f"Suit Table generated in {time.time() - start_time:.2f}s!")
    
    # Honor Suit (字牌)
    print("Generating Honor Table...")
    honor_table = {}
    valid_honor_states = []
    
    def get_valid_honors(idx, current_counts, current_sum):
        if idx == 3:
            valid_honor_states.append(list(current_counts))
            return
        for c in range(5):
            if current_sum + c <= 14:
                current_counts.append(c)
                get_valid_honors(idx + 1, current_counts, current_sum + c)
                current_counts.pop()
                
    get_valid_honors(0, [], 0)
    
    for base_counts in valid_honor_states:
        counts = list(base_counts)
        max_t_no_pair = [-1] * 5
        max_t_pair = [-1] * 5
        
        def dfs_h(idx, g, t, has_pair):
            if g > 4: g = 4
            if t > 4 - g: t = 4 - g
            
            if has_pair:
                if t > max_t_pair[g]: max_t_pair[g] = t
            else:
                if t > max_t_no_pair[g]: max_t_no_pair[g] = t
                
            if idx >= 3:
                return

            if counts[idx] == 0:
                dfs_h(idx + 1, g, t, has_pair)
                return

            if counts[idx] >= 2 and not has_pair:
                counts[idx] -= 2
                dfs_h(idx, g, t, True)
                counts[idx] += 2
                
            if counts[idx] >= 3:
                counts[idx] -= 3
                dfs_h(idx, g + 1, t, has_pair)
                counts[idx] += 3
                
            if counts[idx] >= 2:
                counts[idx] -= 2
                dfs_h(idx, g, t + 1, has_pair)
                counts[idx] += 2

            temp = counts[idx]
            counts[idx] = 0
            dfs_h(idx + 1, g, t, has_pair)
            counts[idx] = temp

        dfs_h(0, 0, 0, False)
        
        encoded = encode_suit(base_counts)
        honor_table[encoded] = (tuple(max_t_no_pair), tuple(max_t_pair))
        
    print(f"Honor Table generated with {len(honor_table)} entries.")

    with open("ruichang_mj_shanten_cache.pkl", "wb") as f:
        pickle.dump({"suit_table": suit_table, "honor_table": honor_table}, f)
        
    print("Saved tables to ruichang_mj_shanten_cache.pkl successfully!")

if __name__ == "__main__":
    main()
