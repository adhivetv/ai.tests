#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU 1
#define ACTIVATION_SOFTMAX 2

__kernel void FullyConnectedLayer(
 __global const float* input,
 __global const float* weights,
 __global const float* biases,
 __global float* output,
int activation_function,
int batch_count,
int input_vector_size,
int output_vector_size)
{
	int element_wise_activation = activation_function == ACTIVATION_SOFTMAX ? ACTIVATION_IDENTITY : activation_function;

	//Get the neuron.
	int neuron_index = get_global_id(0);
	
	//Get the bias for this neuron.
	float bias = biases[neuron_index];
	
	//Select the weights for this neuron.
	 __global const float* weights_vector = &weights[input_vector_size * neuron_index];
	
	//Apply it to each tensor in the input buffer.
	for(int s = 0; s < batch_count; s++)
	{
		//Find the input tensor in the batch.
		 __global const float* input_vector = &input[input_vector_size * s];
		//Find the output tensor in the batch.
		 __global float* output_vector = &output[output_vector_size * s];

		float activated = bias;
	
		for(int i = 0; i < input_vector_size; i++)
		{
			//Compute a dot product.
			activated += input_vector[i] * weights_vector[i];
		}
				
		//Apply activation.
		switch(element_wise_activation)
		{
			case ACTIVATION_IDENTITY:
				break;
			case ACTIVATION_RELU:
				activated = activated > 0.0f ? activated : 0.0f;
				break;
			default:
				return;
		}
		
		//And write it into the output buffer.
		output_vector[neuron_index] = activated;
	}
	
	if(activation_function != ACTIVATION_SOFTMAX)
		return;
	
	//Need all outputs
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	for(int s = 0; s < batch_count; s++)
	{
		 __global float* output_vector = &output[output_vector_size * s];
		
		float max = -FLT_MAX;

		for(int i = 0; i < output_vector_size; i++)
		{
			float v = output_vector[i];
			if(v > max)
				max = v;
		}
		
		//All work items must find the maximum value first.
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		//Compute the exponent for this neuron.
		float v = output_vector[neuron_index] - max; 
		
		float e = exp(v);
		
		//Write the result back...
		output_vector[neuron_index] = e;
		
		//... and synchronize with the rest.
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		//Now sum it all up and divide elementwise by the sum.
		float e_sum = 0.0f;
		
		for(int i = 0; i < output_vector_size; i++)
			e_sum += output_vector[i];
			
		output_vector[neuron_index] /= e_sum;
	}
}