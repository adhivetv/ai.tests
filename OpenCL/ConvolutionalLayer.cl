#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU 1

#define LUT_SIZE 49

float ConvFilter2DPad( __global const float* input_slice,  __global const float* weights, int* lut, int mat_width, int x_from, int x_to, int y_from, int y_to)
{
	float result = 0.0f;

	for(int filter_index = 0, f_y = y_from; f_y < y_to; f_y++)
	{
		int step = mat_width * f_y;
		for(int f_x = x_from; f_x < x_to; f_x++, filter_index++)
		{
			int index = f_x + step;

			float val = input_slice[index];
			float w = weights[lut[filter_index]];

			result += val * w;
		}
	}

	return result;
}

__kernel void ConvolutionalLayer(
 __global const float* input,
 __global const float* weights,
 __global const float* biases,
 __global float* output,
int activation_function,
int batch_count,
int input_depth,
int input_width,
int input_height,
int output_depth,
int output_width,
int output_height,
int stride_x,
int stride_y,
int filter_width,
int filter_height,
int padding_top,
int padding_left)
{
	int out_x = get_global_id(0);
	int out_y = get_global_id(1);
	int filter_index = get_global_id(2);
	
	//Get the bias for this filter.
	float bias = biases[filter_index];
	
	int in_y = out_y * stride_y - padding_top;	
	int in_x = out_x * stride_x - padding_left;

	int lut[LUT_SIZE];
	
	int filter_start_x = 0;
	int filter_start_y = 0;

	int filter_end_x = filter_width;
	int filter_end_y = filter_height;

	int start_x = in_x;
	int start_y = in_y;
	int end_x = filter_width;
	int end_y = filter_height;
	
	if(in_x < 0)
	{	
		filter_start_x -= in_x;

		start_x = 0;
		end_x += in_x;
	}

	if(in_y < 0)
	{
		filter_start_y -= in_y;
		
		start_y = 0;
		end_y += in_y;
	}

	if(in_x + filter_width > input_width)
	{
		filter_end_x -= in_x + filter_width - input_width;

		end_x = filter_end_x;
	}

	if(in_y + filter_height > input_height)
	{
		filter_end_y -= in_y + filter_height - input_height;

		end_y = filter_end_y;
	}
	
	//Initialize the lut.
	for(int f_y = filter_start_y, lut_index = 0; f_y < filter_end_y; f_y++)
	{	
		int step = filter_width * f_y;
		for(int f_x = filter_start_x; f_x < filter_end_x; f_x++, lut_index++)
			lut[lut_index] = f_x + step;
	}
	
	//Select the filter tensor.
	 __global const float* weights_tensor = &weights[filter_width * filter_height * input_depth * filter_index];
	
	//Apply it to each tensor in the input buffer.
	for(int i = 0; i < batch_count; i++)
	{
		//Find the input tensor in the batch.
		 __global const float* input_tensor = &input[input_width * input_height * input_depth * i];
		//Find the output tensor in the batch.
		 __global float* output_tensor = &output[output_width * output_height * output_depth * i];
		//Find the slice in it for this filter.
		 __global float* output_slice = &output_tensor[output_width * output_height * filter_index];
	
		//Initialize with the bias.
		float activated = bias;
	
		//Now apply the filter depthwise.
		for(int d = 0; d < input_depth; d++)
		{
			 __global const float* input_slice = &input_tensor[input_width * input_height * d];
			 __global const float* weights_slice = &weights_tensor[filter_width * filter_height * d];
			
			float conv2d = ConvFilter2DPad(input_slice, weights_slice, lut, input_width, start_x, start_x + end_x, start_y, start_y + end_y);
			//And sum the result.
			activated += conv2d;
		}
					
		//Apply activation.
		switch(activation_function)
		{
			case ACTIVATION_IDENTITY:
				break;
			case ACTIVATION_RELU:
				activated = activated > 0.0f ? activated : 0.0f;
				break;
			default:
				return;
		}
		
		int out_index = out_x + output_width * out_y;
		
		//And write it into the output buffer.
		output_slice[out_index] = activated;
	}
}