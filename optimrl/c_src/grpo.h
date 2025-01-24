#ifndef GRPO_H
#define GRPO_H

#ifdef __cplusplus
extern "C" {
#endif

// Structure to hold batch data for GRPO computation
typedef struct {
    double* log_probs_old;    // Log probabilities from old policy
    double* log_probs_ref;    // Log probabilities from reference policy
    double* rewards;          // Rewards for each action
    int group_size;          // Size of the group
} GRPOBatch;

// Function to compute advantages (standardized rewards)
void compute_advantages(double* rewards, double* advantages, int group_size);

// Main GRPO loss computation function
void grpo_loss(
    GRPOBatch* batch,        // Input batch data
    double* log_probs_new,   // Log probabilities from new policy
    double* loss,            // Output loss value
    double* grad,            // Output gradients
    double epsilon,          // Clipping parameter
    double beta             // KL penalty coefficient
);

#ifdef __cplusplus
}
#endif

#endif /* GRPO_H */