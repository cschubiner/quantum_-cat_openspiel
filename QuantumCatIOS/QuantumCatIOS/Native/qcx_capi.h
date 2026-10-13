// App-owned C declarations for the qcx champion C ABI (qc_capi.cpp).
//
// qc_core.h, qc_agents.h and qc_capi.cpp in this folder are copied verbatim
// from qcx/ by qcx/sync_ios.sh -- never edit them here. This header is
// force-included when compiling qc_capi.cpp (-include qcx_capi.h), so any
// signature drift in the synced C++ is a compile error, not silent UB.
#ifndef QCX_CAPI_H
#define QCX_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/// Length of the int "view" (acting player's information only). Must equal
/// VIEW_LEN in qcx/qc_py.py; qcx/sync_ios.sh fails if they disagree.
#define QCX_VIEW_LEN 84
/// Upper bound on legal actions (4 colours x 9 ranks + paradox); buffers
/// passed to the C ABI are sized generously above this.
#define QCX_MAX_ACTIONS 64

/// Information-set determinized search. Returns the chosen action id
/// (OpenSpiel encoding). values_out receives one mean utility per legal
/// action, in the order of qc_legal_from_view (untouched when only one
/// action is legal).
int qc_decide(const int *view, int worlds, int util, unsigned long long seed, double *values_out);

/// Load the learned rollout policy from the text of a weights file
/// (qcx/weights/*.txt). Returns 1 on success. Call once before deciding.
int qc_load_policy_text(const char *text);
/// Load the learned rollout policy from a weights file path. Returns 1 on success.
int qc_load_policy(const char *path);
/// 1 when a learned rollout policy is loaded.
int qc_policy_loaded(void);
/// Same as qc_decide, but rollouts use the loaded learned play policy
/// (falls back to qc_decide when none is loaded).
int qc_decide_learned(const int *view, int worlds, int util, unsigned long long seed, double *values_out);

/// Legal action ids for the acting player of `view`, written to `out`.
/// Returns the count.
int qc_legal_from_view(const int *view, int *out);

#ifdef __cplusplus
}
#endif

#endif  // QCX_CAPI_H
