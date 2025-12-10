"""AI Assistant Module using rnj-1 for GridOS Dashboard
Provides natural language interface for grid analysis queries
"""

import requests
import json
from typing import Dict, Any, Optional

class GridOSAssistant:
    """AI assistant powered by rnj-1 for power grid analysis"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """Initialize the assistant
        
        Args:
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.ollama_url = f"{ollama_base_url}/api/generate"
        self.model = "rnj-1"
        self.context_window = 32000  # rnj-1 supports 32K tokens
        
        # System prompt for power grid domain
        self.system_prompt = """You are an AI assistant specialized in power grid analysis, 
contingency screening, and GNN-PINN (Graph Neural Networks with Physics-Informed Neural Networks) models.
You help users understand:
- N-1 contingency analysis results
- Risk scores and voltage violations
- Power flow equations and grid topology
- GNN-PINN model predictions

Provide clear, technical explanations suitable for power systems engineers and researchers.
"""
    
    def query(self, user_message: str, grid_context: Optional[Dict[str, Any]] = None, 
              stream: bool = False) -> str:
        """Query the AI assistant
        
        Args:
            user_message: User's question or request
            grid_context: Optional context from current grid analysis
            stream: Whether to stream the response
            
        Returns:
            Assistant's response as string
        """
        # Build the prompt with context
        prompt = self._build_prompt(user_message, grid_context)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096  # Context window for this request
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result.get('response', 'No response generated')
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running with 'ollama serve' and rnj-1 is installed."
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model may be processing a complex query."
        except Exception as e:
            return f"Error querying AI assistant: {str(e)}"
    
    def _build_prompt(self, user_message: str, grid_context: Optional[Dict[str, Any]]) -> str:
        """Build the full prompt with system instructions and context"""
        prompt_parts = [self.system_prompt]
        
        # Add grid context if provided
        if grid_context:
            prompt_parts.append("\n\nCurrent Grid Analysis Context:")
            
            if 'grid_state' in grid_context:
                state = grid_context['grid_state']
                prompt_parts.append(f"Grid: {state.get('name', 'Unknown')} ({state.get('buses', 0)} buses, {state.get('lines', 0)} lines)")
            
            if 'results' in grid_context:
                results = grid_context['results']
                if 'contingencies' in results:
                    top_risks = results['contingencies'][:5]
                    prompt_parts.append(f"\nTop 5 Risky Contingencies:")
                    for i, cont in enumerate(top_risks, 1):
                        prompt_parts.append(f"{i}. {cont['name']}: Risk Score {cont['risk_score']:.3f}, {cont['violations']} violations")
        
        # Add user message
        prompt_parts.append(f"\n\nUser Question: {user_message}\n\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def _handle_stream(self, response) -> str:
        """Handle streaming response from Ollama"""
        full_response = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'response' in chunk:
                    full_response.append(chunk['response'])
        return ''.join(full_response)
    
    def explain_result(self, contingency_result: Dict[str, Any]) -> str:
        """Generate natural language explanation of a contingency result
        
        Args:
            contingency_result: Result from contingency analysis
            
        Returns:
            Natural language explanation
        """
        query = f"""Explain this N-1 contingency analysis result in simple terms:
        
Contingency: {contingency_result.get('name', 'Unknown')}
Type: {contingency_result.get('type', 'N-1')}
Risk Score: {contingency_result.get('risk_score', 0):.3f}
Violations: {contingency_result.get('violations', 0)}
Severity: {contingency_result.get('severity', 0):.2f}

What does this mean for grid stability and what actions should operators consider?"""
        
        return self.query(query)
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and rnj-1 is available
        
        Returns:
            Status dictionary with 'available' and 'message' keys
        """
        try:
            # Check Ollama is running
            response = requests.get(f"{self.ollama_url.replace('/api/generate', '/api/tags')}", timeout=5)
            response.raise_for_status()
            
            # Check if rnj-1 is installed
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            if any('rnj-1' in name for name in model_names):
                return {
                    'available': True,
                    'message': 'rnj-1 is ready',
                    'models': model_names
                }
            else:
                return {
                    'available': False,
                    'message': 'Ollama is running but rnj-1 is not installed. Run: ollama pull rnj-1',
                    'models': model_names
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'available': False,
                'message': 'Ollama is not running. Start it with: ollama serve'
            }
        except Exception as e:
            return {
                'available': False,
                'message': f'Error checking Ollama status: {str(e)}'
            }

# Singleton instance
_assistant_instance = None

def get_assistant() -> GridOSAssistant:
    """Get or create singleton assistant instance"""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = GridOSAssistant()
    return _assistant_instance
