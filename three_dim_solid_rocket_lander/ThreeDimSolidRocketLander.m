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
    %   into screen = -Y direction
    %   out of screen = +Y directions
    %   upwards = -Z direction
    %   downwards = +Z direction
    %   left = -X direction
    %   right = +X direction
    %This ensures a nice right handed coordinate system.
    % Revised: 10-10-2019
    % Copyright 2019 The MathWorks, Inc.
    %we're also assuming that each solid rocket is producing the same amount of
    %thrust on the body.
    %next up,we need to deal with the yaw motion of this orbital dropper. Since
    %we cannot yaw like a quadcopter, which uses torque induced by opposing
    %motors. We will instead add rectangular fins that will be on the side of
    %the obtital drop container. These will interact with the air friction
    %passing over them to introduce some yaw control at high speeds. It is
    %unfortunate that we can only obtain yaw control while moving fast, but it
    %will force the RL agent to figure out it needs to correct the yawing
    %motion while moving quickly.
    
    
    
    %these are the visible properties to the user. We will be taking their 2D
    %counterparts and making them into 3D
    properties
        % Mass of rocket (kg)
        Mass = 2;
        
        % C.G. to top/bottom end (m)
        L1 = 10;
        
        % C.G. to left/right end (m)
        L2 =  5;
        
        % C.G to front/rear end
        L3 = 5;
        
        % Acceleration due to gravity (m/s^2)
        Gravity = 9.806
        
        % Bounds on thrust (N)
        Thrust = 10
        
        % Bounds on angles [degrees]
        Angle = [0 60]
        
        % Sample time (s)
        Ts = 0.1
        
        % State vector
        State = zeros(12,1)
        
        % Last Action values
        LastAction = zeros(5,1)
        
        % Time elapsed during simulation (sec)
        TimeCount = 0
        
        %the aero drag fins to remove yaw motion/stabilize
        FinArea = 0.05; %[m^2]
    end
    
    properties (Hidden)
        % Agent is in continuous mode
        UseContinuousActions = true
        
        % Log for actions and states
        LoggedSignals = cell(2,1)
        
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
            ObservationInfo(1) = rlNumericSpec([13 1 1]);
            ObservationInfo(1).Name = 'states';
            
            % Define action info dimension of the action space
            ActionInfo(1) = rlNumericSpec([5 1 1]);
            ActionInfo(1).Name = 'angles';
            
            % Create environment
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Update action info and initialize states and logs
            updateActionInfo(this);
            this.State = [0 this.L1 0 0 0 0 0 0 0 0 0 0]';
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0 0 0 0]';
            
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
            validateattributes(val,{'numeric'},{'finite','real','vector','numel',5},'','Angle');
            this.Angle = sort(val);
        end
        
        function set.Gravity(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','Gravity');
            this.Gravity = val;
            updateActionInfo(this);
        end
        
        function set.FinArea(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','FinArea');
            this.FinArea = val;
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
            x1 = this.State(1:12);
            [dx1,~,~,action] = dynamics(this,x1, action);
            dx2 = dynamics(this,x1 + ts*dx1, action);
            x = ts/2*(dx1 + dx2) + x1;
            x(7) = atan2(sin(x(7)),cos(x(7)));            % wrap the output angles
            x(8) = atan2(sin(x(8)),cos(x(8)));
            x(9) = atan2(sin(x(9)),cos(x(9)));
            
            
            % Update time count
            this.TimeCount = this.TimeCount + this.Ts;
            
            % Unpack states
            x_  = x(1);
            y_  = x(2) - this.L1;
            z_  = x(3);
            dx_ = x(4);
            dy_ = x(5);
            dz_ = x(6); 
            th_  = x(7);
            ph_  = x(8);
            rh_  = x(9);
            dth_ = x(10);
            dph_ = x(11);
            drh_ = x(12); 
            
            % Determine conditions
            bounds = [100 120-this.L1 100 60 60 60 pi pi pi pi/2 4*pi pi/2];  % bounds on the state-space
            isOutOfBounds = any(abs(x) > bounds(:));
            collision = y_ <= 0.0;
            roughCollision = collision && (dy_ < -1.0 || abs(dx_) > 0.7 || abs(dz_) > 0.7);%...
                %|| abs(dth_) > pi/6 || abs(drh_) > pi/6);
            softCollision = collision && (dy_ >= -1.0 && abs(dx_) <= 0.7 && abs(dz_) <= 0.7);%...
                %&& abs(dth_) < pi/6 && abs(drh_) < pi/6);
            
            % Reward shaping
            distance = sqrt(x_^2 + y_^2 + z_^2) / sqrt(100^2+120^2+100^2);
            speed = sqrt(dx_^2 + dy_^2 + dz_^2) / sqrt(60^2+60^2+60^2);
            s1 = 1 - 0.5 * (sqrt(distance) + sqrt(speed));
            s2 = 0.2 * exp(-th_^2./0.05) + 0.2* exp(-ph_^2./0.05) + 0.2* exp(-rh_^2./0.05);
            shaping = s1+s2;
            reward = shaping;
            %reward = reward + 0.05 * (1 - sum(Action)/2);
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
            thhat = x(7)/bounds(7);  % Normalize to [-1,1]
            phhat = x(8)/bounds(8);
            rhhat = x(9)/bounds(9);
            dthhat = x(10)/bounds(10);
            dphhat = x(11)/bounds(11);
            drhhat = x(12)/bounds(12);
            %so i didn't realise this before, but the observable is not
            %just the state of the object we passed in the landing sensor
            %here as well. I thought is was the time stamp. We're telling
            %it if it succeded or not. I'll need to keep this in mind. 
            landingSensor = 0;
            if roughCollision
                landingSensor = -1;
            end
            if softCollision
                landingSensor = 1;
            end
            nextobs = [xhat; yhat; that; dxhat; dyhat; dthat; thhat; phhat; rhhat; dthhat; dphhat; drhhat; landingSensor];
            
            % Log states and actions
            this.LoggedSignals(1) = {[this.LoggedSignals{1}, this.State(:)]};
            this.LoggedSignals(2) = {[this.LoggedSignals{2}, action(:)]};
            
            % Terminate
            isdone = isOutOfBounds || collision;
        end
        
        %so below is specifying the initial conditions. In my opinion it's
        %important to randomise these. I think it will help the exploration
        %of the agent, but also its overall performance. 
        function obs = reset(this)
            x0 = 0;
            y0 = 100;
            z0 = 0;
            th0 = 0;
            ph0 = 0;
            rh0 = 0;
            if rand
                x0 = -20 + 40*rand;             % vary x from [-20,+20] m
                z0 = -20 + 40*rand;
                th0 = pi/180 * (-45 + 90*rand);  % vary t from [-45,+45] deg
                ph0 = pi/180 * (-45 + 90*rand);
                rh0 = pi/180 * (-45 + 90*rand);
            end
            x = [x0; y0; z0; 0; 0; 0; th0; ph0; rh0; 0; 0; 0];
            
            % Optional: reset to specific state
            hardReset = false;
            if hardReset
                x = [10;100;-45*pi/180;0;0;0;];  % Set the desired initial values here
            end
            
            this.State = x;
            obs = [x; 0]; %this is the landing sensor, not time stamp. 
            this.TimeCount = 0;
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0 0 0 0]';
            
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
                this.ActionInfo(1) = rlNumericSpec([5 1 1],'LowerLimit',LL,'UpperLimit',UL);
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
            
            Tfwd=1;
            Ttwist=1;
            %we are now going to calculate the force from each rocket in the
            %frame of the body.
            norm_factor=10/sqrt(2);
            x_force=norm_factor*(sin(action(1))+sin(action(3))-sin(action(2))-sin(action(4)));
            y_force=norm_factor*(sin(action(4))+sin(action(3))-sin(action(2))-sin(action(1)));
            z_force=norm_factor*(cos(action(1))+cos(action(2))+cos(action(3))+cos(action(4)));
            %now for the torques
            roll_torque=norm_factor*(cos(action(1))+cos(action(3))-cos(action(2))-cos(action(4)));
            pitch_torque=norm_factor*(cos(action(4))+cos(action(3))-cos(action(2))-cos(action(1)));
            yaw_torque=(0.5)*(1.204)*(0.4)*(0.01*sin(action(5)))*cos(action(5))*(x(6).^2);
            
            %get the rotation matrix and stick force into a vector
            force=[x_force,y_force,z_force];
            [rotation_matrix]=calculate_rot_matrix(x);
            inertial_force=force*rotation_matrix;
            
            %these are the parameters which we've talked about before. They
            %can be interfaced by the user. We will be playing with these
            %to checkout the effects of Jupiter gravity on our lander for
            %example.
            L1_ = this.L1;
            L2_ = this.L2;
            L3_ = this.L3;
            m = this.Mass;
            g = this.Gravity;
            
            
            % so I need to decide on the shape of the object. This will
            % change the moments of inertia for each direction. This will
            % be easy to change and should not affect the decisions made
            % by the agent in the future. If so, re train the bastard a
            % bit. We are going to assume it behaves like a flat disc in
            % every direction.
            I1 = 0.5*m*L1_^2;
            I2 = 0.5*m*L2_^2; 
            I3 = 0.5*m*L3_^2; 
            
            %this is our state vector. position and velocity and theta
            x_  = x(1); %#ok<NASGU>
            y_  = x(2);
            z_  = x(3);
            dx_ = x(4);
            dy_ = x(5);
            dz_ = x(6);
            th_ = x(7);
            ph_ = x(8);
            rh_ = x(9);
            dth_= x(10);
            dph_= x(11);
            drh_= x(12);
            
            
            %so here are the forces and the torques being applied to this
            %body. Below, they will be transfered into accelerations by
            %dividing by mass of moments of inertia. The L1,L2,L3 values
            %are giving the dimentions of the rocket lander essentially. I
            %know it is specified as a rectangle above, and yet we're using 
            %moments of inertia from a disc, but it will do for now. 
            Fx = inertial_force(1);
            Fy = inertial_force(2);
            Fz = inertial_force(3);
            My = roll_torque*this.L2;  %rotation about the y-axis is yawing L2 is the rocket radius.(I know this is no correct)
            Mz =  yaw_torque*this.L2;%rotation about the x-axis is the inward and outward          
            Mx =  pitch_torque*this.L3;               %rotation about the z-axis is left and right   
            
            dx = zeros(12,1);
            
            % below are the velocities and accelerations which are caused
            % by the forces here. I'm assuming perfect rotational symmetry
            % for the thing, even though it's not the case
            % treat as "falling" mass
            dx(4) = Fx/m;   %add some jitters here in future. 
            dx(5) = Fy/m; %gravity is in this direction
            dx(6) = Fz/m-g;  %we can maybe add some random jitters here in future
            dx(10)= Mx/I1;
            dx(11)= My/I1;
            dx(12)= Mz/I2; 
            dx(1) = dx_;
            dx(2) = dy_;
            dx(3) = dz_;
            dx(7) = dth_;
            dx(8) = dph_;
            dx(9) = drh_;
           
        end
        
    end
    
end