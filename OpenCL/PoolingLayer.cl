void MaxFilter( __global const float* input_slice,  float* value, int* pooled, int mat_width, int x_from, int x_to, int y_from, int y_to)
{
	float max_elem = -FLT_MAX;
	int max_index = 0;
	
	for(int filter_index = 0, f_y = y_from; f_y < y_to; f_y++)
	{
		int step = mat_width * f_y;
		for(int f_x = x_from; f_x < x_to; f_x++, filter_index++)
		{
			int index = f_x + step;
			float val = input_slice[index];

			if(val > max_elem)
			{
				max_elem = val;
				max_index = filter_index;
			}
		}
	}
	
	*value = max_elem;
	*pooled = max_index;
}

__kernel void MaxPool(
 __global const float* input, 
 __global float* output, 
 __global int* pooled,
int batch_count,
int tensor_depth,
int input_width,
int input_height,
int output_width,
int output_height,
int stride_x, 
int stride_y, 
int filter_width, 
int filter_height)
{
	int out_x = get_global_id(0);
	int out_y = get_global_id(1);
	int out_z = get_global_id(2);
	
	int in_x = out_x * stride_x;
	int in_y = out_y * stride_y;

	for(int i = 0; i < batch_count; i++)
	{
		float value = 0.0f;
		int pooled_index = 0;
		
		 __global const float* input_tensor = &input[input_width * input_height * tensor_depth * i];
		 __global float* output_tensor = &output[output_width * output_height * tensor_depth * i];
		 __global int* pooled_tensor = &pooled[output_width * output_height * tensor_depth * i];	
		
		 __global const float* input_slice = &input_tensor[input_width * input_height * out_z];
		 __global float* output_slice = &output_tensor[output_width * output_height * out_z];
		 __global int* pooled_slice = &pooled_tensor[output_width * output_height * out_z];	
		
		MaxFilter(input_slice, &value, &pooled_index, input_width, in_x, in_x + filter_width, in_y, in_y + filter_height);
		
		int out_index = out_x + output_width * out_y;
		
		output_slice[out_index] = value;
		pooled_slice[out_index] = pooled_index;
	}
}