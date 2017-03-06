#ifndef __GPUANALYSIS__
#define __GPUANALYSIS__

void analysis_GPU(float *h_output_list, size_t *list_pos, size_t max_list_size, float *h_peak_list, size_t *peak_pos, size_t max_peak_size, int i, float tstart, int t_processed, int inBin, int outBin, int *maxshift, int max_ndms, int *ndms, float cutoff, float sigma_constant, int max_boxcar_width, float *output_buffer, float *dm_low, float *dm_high, float *dm_step, float tsamp);

#endif





