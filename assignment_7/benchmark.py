"""
API Benchmark and Testing Script
This script tests the FastAPI deployment and measures performance metrics
"""

import requests
import time
import json
import statistics
from typing import List, Dict
from datetime import datetime
import concurrent.futures

# API Configuration
API_BASE_URL = "http://localhost:8000"

class APIBenchmark:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.results = {
            "health_checks": [],
            "single_requests": [],
            "batch_requests": [],
            "concurrent_requests": []
        }
    
    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f" {text}")
        print("="*70)
    
    def test_health_check(self) -> bool:
        """Test the health endpoint"""
        self.print_header("Testing Health Check Endpoint")
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_model_info(self):
        """Test the model info endpoint"""
        self.print_header("Testing Model Info Endpoint")
        try:
            response = requests.get(f"{self.base_url}/model-info")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
    
    def single_generation_test(self, prompt: str, iterations: int = 5) -> List[float]:
        """Test single text generation multiple times"""
        self.print_header(f"Single Generation Test ({iterations} iterations)")
        print(f"Prompt: '{prompt}'")
        
        latencies = []
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_length": 100,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.9,
                        "num_return_sequences": 1
                    }
                )
                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"\nIteration {i+1}:")
                    print(f"  Status: Success")
                    print(f"  Total Latency: {latency:.3f}s")
                    print(f"  Server Generation Time: {result['generation_time']:.3f}s")
                    print(f"  Generated Text (first 100 chars): {result['generated_texts'][0][:100]}...")
                else:
                    print(f"\nIteration {i+1}: Failed (Status {response.status_code})")
                
            except Exception as e:
                print(f"\nIteration {i+1}: Error - {e}")
        
        # Calculate statistics
        if latencies:
            print(f"\n{'='*70}")
            print("Performance Statistics:")
            print(f"  Average Latency: {statistics.mean(latencies):.3f}s")
            print(f"  Median Latency: {statistics.median(latencies):.3f}s")
            print(f"  Min Latency: {min(latencies):.3f}s")
            print(f"  Max Latency: {max(latencies):.3f}s")
            if len(latencies) > 1:
                print(f"  Std Deviation: {statistics.stdev(latencies):.3f}s")
        
        return latencies
    
    def batch_generation_test(self):
        """Test batch generation"""
        self.print_header("Batch Generation Test")
        
        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "Machine learning enables",
            "Cloud computing provides"
        ]
        
        print(f"Testing with {len(prompts)} prompts")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/generate-batch",
                params={"max_length": 100},
                json=prompts
            )
            
            end_time = time.time()
            total_latency = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nStatus: Success")
                print(f"Total Latency: {total_latency:.3f}s")
                print(f"Server Total Time: {result['total_time']:.3f}s")
                print(f"Average Time per Prompt: {result['average_time_per_prompt']:.3f}s")
                print(f"\nGenerated Texts:")
                for i, item in enumerate(result['results'], 1):
                    print(f"\n  {i}. Prompt: '{item['prompt']}'")
                    print(f"     Generated (first 80 chars): {item['generated_text'][:80]}...")
                    print(f"     Time: {item['generation_time']:.3f}s")
            else:
                print(f"Failed (Status {response.status_code})")
        
        except Exception as e:
            print(f"Error: {e}")
    
    def concurrent_requests_test(self, num_requests: int = 10):
        """Test concurrent requests"""
        self.print_header(f"Concurrent Requests Test ({num_requests} requests)")
        
        def make_request(i: int) -> Dict:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": f"Test prompt {i}:",
                        "max_length": 50,
                        "temperature": 0.7
                    }
                )
                end_time = time.time()
                
                return {
                    "request_id": i,
                    "status_code": response.status_code,
                    "latency": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "request_id": i,
                    "error": str(e),
                    "success": False
                }
        
        print(f"Sending {num_requests} concurrent requests...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        print(f"\nResults:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Successful Requests: {len(successful)}/{num_requests}")
        print(f"  Failed Requests: {len(failed)}")
        
        if successful:
            latencies = [r["latency"] for r in successful]
            print(f"\nLatency Statistics:")
            print(f"  Average: {statistics.mean(latencies):.3f}s")
            print(f"  Median: {statistics.median(latencies):.3f}s")
            print(f"  Min: {min(latencies):.3f}s")
            print(f"  Max: {max(latencies):.3f}s")
            print(f"  Throughput: {len(successful)/total_time:.2f} requests/second")
    
    def stress_test(self, duration_seconds: int = 30):
        """Run stress test for specified duration"""
        self.print_header(f"Stress Test (Duration: {duration_seconds}s)")
        
        print("Sending continuous requests...")
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_count = 0
        success_count = 0
        latencies = []
        
        while time.time() < end_time:
            try:
                req_start = time.time()
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": "Test",
                        "max_length": 30
                    },
                    timeout=10
                )
                req_end = time.time()
                
                request_count += 1
                if response.status_code == 200:
                    success_count += 1
                    latencies.append(req_end - req_start)
                
                # Print progress every 5 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    print(f"  Progress: {int(elapsed)}s - Requests: {request_count}, Success: {success_count}")
                
            except Exception as e:
                request_count += 1
                print(f"  Request failed: {e}")
        
        total_duration = time.time() - start_time
        
        print(f"\nStress Test Results:")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Total Requests: {request_count}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {request_count - success_count}")
        print(f"  Success Rate: {(success_count/request_count)*100:.2f}%")
        print(f"  Average Throughput: {request_count/total_duration:.2f} requests/second")
        
        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Average: {statistics.mean(latencies):.3f}s")
            print(f"  Median: {statistics.median(latencies):.3f}s")
            print(f"  Min: {min(latencies):.3f}s")
            print(f"  Max: {max(latencies):.3f}s")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print(" GENERATIVE AI API - COMPREHENSIVE BENCHMARK")
        print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Test 1: Health Check
        if not self.test_health_check():
            print("\n❌ Health check failed! API may not be running.")
            print("Please ensure the API is running at", self.base_url)
            return
        
        # Test 2: Model Info
        self.test_model_info()
        
        # Test 3: Single Generation
        self.single_generation_test("Artificial intelligence is", iterations=5)
        
        # Test 4: Batch Generation
        self.batch_generation_test()
        
        # Test 5: Concurrent Requests
        self.concurrent_requests_test(num_requests=10)
        
        # Test 6: Stress Test (optional, commented out by default)
        # self.stress_test(duration_seconds=30)
        
        print("\n" + "="*70)
        print(" BENCHMARK COMPLETED")
        print("="*70)

def main():
    """Main function"""
    print("Starting API Benchmark...")
    print(f"API URL: {API_BASE_URL}")
    print("\nMake sure the API is running before starting the benchmark!")
    print("Start the API with: python app.py")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nBenchmark cancelled.")
        return
    
    benchmark = APIBenchmark()
    benchmark.run_full_benchmark()

if __name__ == "__main__":
    main()
