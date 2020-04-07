The purpose of this project is primarily to learn how to create a MATLAB environment for reinforcement learning and become familiar with the RL library that MATLAB 
constructed. I have done this by using MATLAB's RocketLander class which simulates a rocket landing. I have replaced the 'liquid' simulated rockets with solid rockets
that change orientation relative to the body and therefore can change the rotation and the net force on the body. 
	I've also changed the agent from running on a discreet action space, to running on a continuous one. This just organically makes sense as we cannot turn a solid rocket
boosted on and off as we please, once it stars, we burn all the way to the end. 
	The second goal of this particular project is to observe the effects that poor engineering can have on a body and find out if our agent is able to compensate and where
we can find the limits of an RL agent. These are the questions that would be interesting to answer:

1. How much angular momentum can the RL agent deal with if it's been trained on a small initial angular momentum? 
2. How much mass can be added before the agent cannot create enough thrust to land correctly. 
3. How little thrust can the rockets have before we encounter issues?
4. What happens id we have centers of mass which are very high? will the RL agent be able to compensate for this?

All quesetions that will help define some limits of our agents and indicate when more training is required.
