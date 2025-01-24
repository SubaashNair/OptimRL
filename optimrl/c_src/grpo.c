#include "grpo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function to compute robust statistics
void compute_reward_stats(double* rewards, int group_size, double* out_mean, double* out_std) {
    // First pass: compute true mean with stable summation
    double mean = 0.0;
    for (int i = 0; i < group_size; i++) {
        mean += rewards[i];
    }
    mean /= (double)group_size;
    
    // Second pass: compute variance with stable computation
    double M2 = 0.0;
    for (int i = 0; i < group_size; i++) {
        double delta = rewards[i] - mean;
        M2 += delta * delta;
    }
    
    // Compute standard deviation with consistent normalization
    double variance = M2 / (double)(group_size); // Remove Bessel's correction
    double std = sqrt(variance + 1e-8);
    
    // Ensure non-zero std for numerical stability
    if (std < 1e-8) {
        std = 1.0;
    }
    
    *out_mean = mean;
    *out_std = std;
}

void compute_advantages(double* rewards, double* advantages, int group_size) {
    double mean, std;
    compute_reward_stats(rewards, group_size, &mean, &std);
    
    // Compute advantages with consistent normalization
    for (int i = 0; i < group_size; i++) {
        advantages[i] = (rewards[i] - mean) / std;
    }
}

void grpo_loss(
    GRPOBatch* batch,
    double* log_probs_new,
    double* loss,
    double* grad,
    double epsilon,
    double beta
) {
    int G = batch->group_size;
    double* advantages = (double*)malloc(G * sizeof(double));
    compute_advantages(batch->rewards, advantages, G);
    
    // DEBUG message to confirm which code is running
    printf("DEBUG: Using the new grpo_loss with post-loop division!\n");
    fflush(stdout);

    // Initialize accumulators for loss and intermediate values
    double total_surrogate = 0.0;
    double total_kl = 0.0;
    
    // Process each sample in the batch
    for (int i = 0; i < G; i++) {
        // Check for identical policies with improved numerical precision
        double log_ratio = log_probs_new[i] - batch->log_probs_old[i];
        int is_identical = fabs(log_ratio) < 1e-7;
        
        // Compute probability ratios with careful numerical handling
        double ratio = is_identical ? 1.0 : exp(log_ratio);
        double clipped_ratio = ratio;
        
        if (!is_identical) {
            clipped_ratio = fmax(fmin(ratio, 1.0 + epsilon), 1.0 - epsilon);
        }
        
        // Compute surrogate loss
        double surrogate1 = ratio * advantages[i];
        double surrogate2 = clipped_ratio * advantages[i];
        double min_surrogate = fmin(surrogate1, surrogate2);
        
        // Compute KL divergence term
        double kl_term = 0.0;
        if (!is_identical) {
            double log_diff = log_probs_new[i] - batch->log_probs_ref[i];
            // Based on some KL approx expression or direct expansions
            kl_term = exp(batch->log_probs_ref[i] - log_probs_new[i]) - log_diff - 1.0;
        }
        
        // Accumulate terms
        total_surrogate += min_surrogate;
        total_kl += kl_term;
        
        // Compute gradient
        if (is_identical) {
            // If identical, gradient is effectively zero
            grad[i] = 0.0;
        } else {
            double policy_grad = (surrogate1 < surrogate2) 
                ? ratio * advantages[i] 
                : clipped_ratio * advantages[i];
            
            double kl_grad = beta * (exp(batch->log_probs_ref[i] - log_probs_new[i]) + 1.0);
            
            // -------------------------------
            // Remove the division by G here:
            // grad[i] = (policy_grad - kl_grad) / G;   // OLD
            // NEW:
            grad[i] = (policy_grad - kl_grad);
            // -------------------------------
        }
    }
    
    // ----------------------------------------
    // After the loop, divide each grad[i] by G
    for (int i = 0; i < G; i++) {
        grad[i] /= (double) G;
    }
    // ----------------------------------------
    
    // Compute final loss with consistent scaling
    *loss = -(total_surrogate - beta * total_kl) / G;
    
    free(advantages);
}
