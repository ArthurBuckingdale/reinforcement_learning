classdef ThreeDimSolidRocketLander < rl.env.MATLABEnvironment
%This class is not written from scratch and parts of it are borrowed from
%the folks at matlab. until now, this rocket land is just in two
%dimentions. We not want to scale this up to three since we do not live in
%a 2D world and this 2d rocket lander is essentially useless. In order to
%do with we will leverage the SolidRocketLander class and up the number of
%dimentions to three. This adds a shit load of problems, because we need to
%deal with a whole new array of forces and twisting will now be a big deal.
%to cope with three dimensions we are going to add another two solid rocket
%boosters to our 'orbital drop' container. Imagine a cube with a solid
%rocket booster strapped to each corner these solid rockets can shift
%angles and they are going to extend outward following the diagonal of the
%cube. just think of this being similar in control scheme to a drone with
%an x-frame pattern. This is exactly what we're attempting to build, but
%it's an orbital drop rocket lander. Below is a drawing of the cube and how
%we will go about assigning the actions to each. action number is labelled
%which will be going into where. 
%3              4
%|-------------|
%|             |
%|             |
%|             |
%|             | 
%|-------------|
%1              2
%   into screen = -Z direction
%   out of screen = +Z directions
%   upwards = +Y direction 
%   downwards = -Y direction
%   left = -X direction
%   right = +X direction 
%This ensures a nice right handed coordinate system. 
% Revised: 10-10-2019
% Copyright 2019 The MathWorks, Inc.
%we're also assuming that each solid rocket is producing the same amount of
%thrust on the body. 


%these are the visible properties to the user. We will be taking their 2D
%counterparts and making them into 3D
    properties
        % Mass of rocket (kg)
        Mass = 1;
        
        % C.G. to top/bottom end (m)
        L1 = 10;
        
        % C.G. to left/right end (m)
        L2 =  5;
        
        % C.G to front/rear end
        L3 = 5
        
        % Acceleration due to gravity (m/s^2)
        Gravity = 9.806
        
        % Bounds on thrust (N)
        Thrust = 10
        
        % Bounds on angles [degrees]
        Angle = [0 90]
                
        % Sample time (s)
        Ts = 0.1
        
        % State vector
        State = zeros(12,1)
        
        % Last Action values
        LastAction = zeros(4,1)
        
        % Time elapsed during simulation (sec)
        TimeCount = 0
    end
    
    properties (Hidden)        
        % Agent is in continuous mode
        UseContinuousActions = true 
        
        % Log for actions and states
        LoggedSignals = cell(4,1)
        
        % Flags for visualization
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false        
    end
    
    properties (Transient,Access = private)
        Visualizer = []
    end
    
%the methods below are part of the obligatory functions which are
%required for the environment. They perform things like the following:
%1. constructors for the environment, defining the action space and the
%observation space(observations are what's coming in from our sensors and
%the action space is what's being output from our RL agent). 
%2. Here are the
%functions which will validate and set our parameters that we place in. 
%3.It also contains both the soft and hard reset values that are needed for the
%episodes to initialise after they. 
%4.also contains functions which will
%allow us to plot and sim the environment where we can see the agent
%interacting with the environment. 
%5. contains the step function. this is what is taking in the time and
%action parameters to give the feedback for reward and if we've failed the
%environment parameters or not. 
%6.the reward is calculated here and this is where we want to go and modify
%it. 
%last note these are the public 
    methods
        
        function this = ThreeDimSolidRocketLander(ActionInfo) % here is our constructor function
            
            % Define observation info dimension of the observation space
            ObservationInfo(1) = rlNumericSpec([12 1 1]);
            ObservationInfo(1).Name = 'states';
            
            % Define action info dimension of the action space
            ActionInfo(1) = rlNumericSpec([4 1 1]);
            ActionInfo(1).Name = 'angles';
            
            % Create environment
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Update action info and initialize states and logs
            updateActionInfo(this);
            this.State = [0 this.L1 0 0 0 0 0 0 0 0 0 0]';
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0 0 0]';
            
        end
        
        % below is the set of functions which are validating and setting
        % the paremeters from properties
        function set.UseContinuousActions(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','UseContinuousActions');
            this.UseContinuousActions = logical(val);
            updateActionInfo(this);
        end
        
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',12},'','State');
            this.State = state(:);
            notifyEnvUpdated(this);
        end
        
        function set.Mass(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Mass');
            this.Mass = val;
        end
        
        function set.L1(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L1');
            this.L1 = val;
        end
        
        function set.L2(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L2');
            this.L2 = val;
        end
        
        function set.L3(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L3');
            this.L3 = val;
        end
        
        function set.Thrust(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Thrust');
            this.Thrust = val;
        end
        
        function set.Angle(this,val)
            validateattributes(val,{'numeric'},{'finite','real','vector','numel',4},'','Angle');
            this.Angle = sort(val);
        end
        
        function set.Gravity(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','Gravity');
            this.Gravity = val;
            updateActionInfo(this);
        end
        
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
        
        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = RocketLanderVisualizer(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            % Reset Visualizations
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end
        
        %so this right here is our step function is is what's taking in the
        %actions and the physical paramaters and stepping forwards in time
        %to obtain results of a new state. 
        function [nextobs,reward,isdone,loggedSignals] = step(this,Action)
            
            loggedSignals = [];
            
            % Scale up the actions
            action = Action .* this.Angle(2);
            
            % Trapezoidal integration
            ts = this.Ts;
            x1 = this.State(1:6);
            [dx1,~,~,action] = dynamics(this,x1, action);
            dx2 = dynamics(this,x1 + ts*dx1, action);
            x = ts/2*(dx1 + dx2) + x1;
            x(3) = atan2(sin(x(3)),cos(x(3)));   % wrap the output angle
            
            % Update time count
            this.TimeCount = this.TimeCount + this.Ts;
            
            % Unpack states
            x_  = x(1);
            y_  = x(2) - this.L1;
            t_  = x(3); 
            dx_ = x(4);
            dy_ = x(5);
            dt_ = x(6); %#ok<NASGU>

            % Determine conditions
            bounds = [100 120-this.L1 pi 60 60 pi/2];  % bounds on the state-space
            isOutOfBounds = any(abs(x) > bounds(:));
            collision = y_ <= 0;
            roughCollision = collision && (dy_ < -0.5 || abs(dx_) > 0.5);
            softCollision = collision && (dy_ >= -0.5 && abs(dx_) <= 0.5);            
            
            % Reward shaping
            distance = sqrt(x_^2 + y_^2) / sqrt(100^2+120^2);
            speed = sqrt(dx_^2 + dy_^2) / sqrt(60^2+60^2);
            s1 = 1 - 0.5 * (sqrt(distance) + sqrt(speed));
            s2 = 0.5 * exp(-t_^2./0.05);
            shaping = s1+s2;
            reward = shaping;
            reward = reward + 0.05 * (1 - sum(Action)/2);
            reward = reward + 10000 * softCollision;
            
            % Set the states and last action
            this.State = x;
            this.LastAction = Action(:);
            
            % Set the observations
            xhat = x(1)/bounds(1);  % Normalize to [-1,1]
            yhat = x(2)/bounds(2);
            that = x(3)/bounds(3); 
            dxhat = x(4)/bounds(4); 
            dyhat = x(5)/bounds(5);
            dthat = x(6)/bounds(6);
            landingSensor = 0;
            if roughCollision
                landingSensor = -1;
            end
            if softCollision
                landingSensor = 1;
            end
            nextobs = [xhat; yhat; that; dxhat; dyhat; dthat; landingSensor];
            
            % Log states and actions
            this.LoggedSignals(1) = {[this.LoggedSignals{1}, this.State(:)]};
            this.LoggedSignals(2) = {[this.LoggedSignals{2}, action(:)]};
            
            % Terminate
            isdone = isOutOfBounds || collision;
        end
        
        function obs = reset(this)
            x0 = 0;
            y0 = 100;
            t0 = 0;
            if rand
                x0 = -20 + 40*rand;             % vary x from [-20,+20] m
                t0 = pi/180 * (-45 + 90*rand);  % vary t from [-45,+45] deg
            end
            x = [x0; y0; t0; 0; 0; 0];
            
            % Optional: reset to specific state
            hardReset = true;
            if hardReset
                x = [10;100;-45*pi/180;0;0;0;];  % Set the desired initial values here
            end
            
            this.State = x;
            obs = [x; 0];
            this.TimeCount = 0;
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0]';   
            
            %reset(this.Visualizer);            
        end
        
    end
    
    %so these are the private access functions. These are the ones which I
    %understand will be changing the action and computing the dynamics of
    %the object to feed to the inputs of the NN. In order for this thing to
    %land itself, it'll need to know its acceleration, velocity how far
    %from the ground it is and its orientation relative to the ground for
    %example. 
    
    methods (Access = private)
        
        %this is the function which is taking in the state of the object
        %and updating the action information that it's going to take. This
        %particular version can return two different types. One is
        %continuous and the other is discreet. Since we're going to use angles
        %we definitely require a discreet space for this, simply because
        %more range is just better. 
        function updateActionInfo(this)
            LL = 0; 
            UL = 1;
            if this.UseContinuousActions
                this.ActionInfo(1) = rlNumericSpec([2 1 1],'LowerLimit',LL,'UpperLimit',UL);
            else
                ML = (UL - LL)/2 + LL;
                els = {...
                    [LL;LL],...  % do nothing
                    [LL;ML],...  % fire right (med)  
                    [LL;UL],...  % fire right (max)
                    [ML;LL],...  % fire left (med)
                    [ML;ML],...  % fire left (med) + right (med)
                    [ML;UL],...  % fire left (med) + right (max)
                    [UL;LL],...  % fire left (max)
                    [UL;ML],...  % fire left (max) + right (med)
                    [UL;UL] ...  % fire left (max) + right (max)
                    }';
                this.ActionInfo = rlFiniteSetSpec(els);
            end
            this.ActionInfo(1).Name = 'angles';
        end
        
        %this is the function which is computing the dynamics. The MATLAB
        %folks have documented it very well so no extra commentary is
        %needed from me to explain this. 
        function [dx,Tfwd,Ttwist,action] = dynamics(this,x,action)
        % DYNAMICS calculates the state derivatives of the robot.
        %
        % m     = Mass of the rocket (kg)
        % g     = Acceleration due to gravity (m/s^2)
        % L1    = Radius of the rocket (m)
        % L2    = Distance of thrusters from the rocket's center of mass (m)
        % I     = Moment of inertia of the rocket (kg-m^2)
        % x     = x coordinate of rocket's center of mass (m)
        % y     = y coordinate of rocket's center of mass (m)
        % z     = Z coordinate of rocket's center of mass (m) 
        % t     = angle of the rocket w.r.t. y-axis (rad, counterclockwise positive)
        % p     = angle of the rocket w.r.t. x axis (rad, counterclockwise positive)
        % r     = angle of the rocket w.r.t. z-axis (rad, counterclockwise positive)
        % dx    = x velocity of rocket's center of mass (m/s)
        % dy    = y velocity of rocket's center of mass (m/s)
        % dz    = z velocity of rocket's center of mass (m/s)
        % dt    = angular velocity of rocket (rad/s)
        % dp    = angular velocity of rocket (rad/s)
        % dr    = angular velocity of rocket (rad/s)
        % k     = Ground stiffness coefficient (N/m)
        % c     = Ground damping coefficient (N/m/s)
        % T1,T2 = Thrust values (N)
        %
        % Fx    = -sin(t)*(T1 + T2)
        % Fy    = cos(t)*(T1 + T2)
        % Mz    = (T2 - T1) * L2
        %
        % In air:
        %       m*ddx =  Fx
        %       m*ddy =  Fy - m*g
        %       I*ddt = Mz
        % On ground:
        %       (I + m*L1^2)*ddx = Fx*L1^2 - Mz*L1   (rolling disc)
        %       m*ddy + k*(y-L1) + c*dy = Fy - m*g
        %       L1*ddt = -ddx
        
            
        %so we now need to get into all the forces on this body, they
           %will increase a lot now that we're in 3 dimensions.
           %recall that this is very similar to a quadcopter drone. We have
           %less control directions than directions. Which means we need to
           %induce a certain amount of roll to strafe for example. A
           %certain amount of tilt to move forwards.
           
           %
            Tupdwards   =   this.Thrust.*cosd(action(2)) + this.Thrust.*cosd(action(1));
            Tforwards   =   this.Thrust.*cosd(action(2)) + this.Thrust.*cosd(action(1));
            Trightwards = this.Thrust.*cosd(action(2)) + this.Thrust.*cosd(action(1));            
            Ttheta = this.Thrust.*sind(action(2)) - this.Thrust.*sind(action(1));
            Tphi = this.Thrust.*sind(action(2)) - this.Thrust.*sind(action(1));
            Trho = this.Thrust.*sind(action(2)) - this.Thrust.*sind(action(1));
            
            %these are the parameters which we've talked about before. They
            %can be interfaced by the user. We will be playing with these
            %to checkout the effects of Jupiter gravity on our lander for
            %example. 
            L1_ = this.L1;
            L2_ = this.L2;
            m = this.Mass;
            g = this.Gravity;
            % inertia for a flat disk inertia in the rotation equivalent to
            % mass i.e it resists changes in motion. 
            I = 0.5*m*L1_^2;
            
            %this is our state vector. position and velocity and theta
            x_  = x(1); %#ok<NASGU>
            y_  = x(2); 
            t_  = x(3);
            dx_ = x(4);
            dy_ = x(5);
            dt_ = x(6);
            
            %so this is the various forces acting on our object. We can see
            %the components for each side. the sin and cos of theta are
            %telling us where our thrust is pointing since the body can
            %roll. The rolling force is also added here and is affected by
            %the center of gravity. We will show by changing this center of
            %gravity the effect is has on our ability to land something.
            %This is the section where we're introducing the angle of the
            %spaccraft relative to the ground. and is the third angle used
            %in this calculation. 
            Fx = -sin(t_) * Tfwd;
            Fy =  cos(t_) * Tfwd;
            Mz =  Ttwist * L2_  ;
            
            dx = zeros(6,1);
            
            % ground penetration. 
            yhat = y_ - L1_;
            
            
           %so this is interesting, we are changing the regime as we
           %approach the ground. This is quite interesting. I think it
           %makes sense, but also just very interesting. I'll need to
           %observe the effects of this. It might just be there to level
           %out the rocket once its CG is below the ground level. 
            if yhat < 0
                % "normalized" for mass
                k = 1e2;
                c = 1e1;
                % treat as rolling wheel (1 DOF is lost between x and theta)
                dx(4) = (Fx*L1_^2 - Mz*L1_)/(I + m*L1_^2);
                dx(5) = Fy/m - g - k*yhat - c*dy_;
                dx(6) = -dx(4,1)/L1_;
                dx(1) = dx_;
                dx(2) = dy_;
                dx(3) = -dx(1,1)/L1_;
            else
                % treat as "falling" mass
                dx(4) = Fx/m;
                dx(5) = Fy/m - g;
                dx(6) = Mz/I; %the rotational equivalent of an acceleration
                dx(1) = dx_;
                dx(2) = dy_;
                dx(3) = dt_;
            end            
            
        end
        
    end
    
end