export Actor
using Flux



Struct Actor
   state_size::Int8
   action_size::Int8
   layer_size = Int8[5,6,4]
   learning_rate::Float64=0.001
   batch_size::Int8=6
   tau::Float64= 0.01
   """ Connstructor for the Actor Network
     state_size:An integer denoting the dimensionality of the states
            in the current problem
     action_size: An integer denoting the dimensionality of the
            actions in the current problem
      hidden_units: An iterable defining the number of hidden units in
            each layer
      learning_rate: A fload denoting the speed at which the network
           will learn. default
      batch_size: An integer denoting the batch size 
      tau: A flot denoting the rate at which the target model will
            track the main model. Formally, the tracking function is defined as:
              target_weights = tau * main_weights + (1 - tau) * target_weights """

   function generate_model(s::Int8=state_size,I::Array{Float64})
       """Generates the model based on the hyperparameters defined in the constructor"""
       input_layer = I
       weights1 = rand(5,6)
       weights2 = rand(6,4)  
       model = Flux.chain(Dense(param(weights1),relu),Dense(param(weights2),sigmoid));
       return model(I) ,weights1, I 
    end
    function train(states::Array{Float64},action_gradients::Array{Float64})
         """Updates the weights of the main network
            states: The states of the input to the network
            action_gradients: The gradients of the actions to update the
            network"""
         model1,model1_weight,model1_input = generate_model()
         for i = 0:100
            model1_weight .-= learning_rate .*action_gradients
            function train_target_model(m::Array({Float64},5,6))
            """ Updates the weights of the target network to slowly track the main
        network.
        The speed at which the target network tracks the main network is
        defined by tau, given in the constructor to this class. Formally,
        the tracking function is defined as:
            target_weights = tau * main_weights + (1 - tau) * target_weights"""
                main_weights = m
                target_model,target_model_weight, target_state = generate_model()
                target_weights = target_model_weight
                target_weights = tau.*main_weight .+ (1 - tau).*target_weight
                target_model_weight = target_weights
             end
    end
