import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_reflection_steps = config.num_reflection_steps
        
        self.thought_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        self.thought_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            ) for _ in range(self.num_reflection_steps - 1)
        ])
        
        self.final_projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        original_input = x
        
        # Initial thought generation
        thought = self.thought_generator(x)
        
        # Multiple steps of thought refinement
        for refinement_layer in self.thought_refinement:
            combined_input = torch.cat([x, thought], dim=-1)
            thought = refinement_layer(combined_input)
            thought = thought + x  # Residual connection
        
        # Final reflection
        final_combined = torch.cat([original_input, thought], dim=-1)
        final_output = self.final_projection(final_combined)
        
        # Residual connection and normalization
        output = self.layer_norm(original_input + self.dropout(final_output))
        
        return output