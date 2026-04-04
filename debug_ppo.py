import torch
import torch.multiprocessing as mp
import time
from resnet_model import MahjongActorCritic
from train_sl import get_device

def simple_worker(worker_id, request_queue, response_queue):
    print(f"Worker {worker_id} starting...")
    device = torch.device("cpu") # 调试阶段先用 CPU
    state = torch.randn(14, 30).numpy()
    mask = torch.ones(30).numpy()
    
    request_queue.put((worker_id, state, mask))
    print(f"Worker {worker_id} sent request.")
    
    action, log_prob, val = response_queue.get()
    print(f"Worker {worker_id} received: action={action}, val={val}")

def simple_inference(request_queue, response_queues):
    print("Inference Server starting...")
    device = get_device()
    model = MahjongActorCritic().to(device)
    model.eval()
    print("Inference Server ready.")
    
    while True:
        try:
            wid, state, mask = request_queue.get(timeout=5)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value = model(state_t, legal_mask=mask_t)
                action = torch.argmax(logits, dim=-1).item()
                
            response_queues[wid].put((action, 0.0, value.item()))
            print(f"Inference Server processed worker {wid}")
        except Exception as e:
            print(f"Inference Server timeout or error: {e}")
            break

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    req_q = mp.Queue()
    res_qs = [mp.Queue() for _ in range(1)]
    
    inf_p = mp.Process(target=simple_inference, args=(req_q, res_qs))
    inf_p.start()
    
    time.sleep(2)
    
    wrk_p = mp.Process(target=simple_worker, args=(0, req_q, res_qs[0]))
    wrk_p.start()
    
    wrk_p.join()
    inf_p.terminate()
    print("Debug test finished.")
