classdef RocketLander < rl.env.MATLABEnvironment
% ROCKETLANDER environment models a 3-DOF disc-shaped rocket with mass. 
%
% The rocket has two thrusters for forward and rotational motion. 
% Gravity acts vertically downwards, and there are no aerodynamic 
% drag forces.
%
% Revised: 10-10-2019
% Copyright 2019 The MathWorks, Inc.

    properties
        % Mass of rocket (kg)
        Mass = 1;
        
        % C.G. to top/bottom end (m)
        L1 = 10;
        
        % C.G. to left/right end (m)
        L2 =  5;      
        
        % Acceleration due to gravity (m/s^2)
        Gravity = 9.806
        
        % Bounds on thrust (N)
        ThrustLimits = [0 10]
                
        % Sample time (s)
        Ts = 0.1
        
        % State vector
        State = zeros(6,1)
        
        % Last Action values
        LastAction = zeros(2,1)
        
        % Time elapsed during simulation (sec)
        TimeCount = 0
    end
    
    properties (Hidden)        
        % Agent is in continuous mode
        UseContinuousActions = false 
        
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
    
    methods
        
        function this = RocketLander(ActionInfo)
            
            % Define observation info
            ObservationInfo(1) = rlNumericSpec([7 1 1]);
            ObservationInfo(1).Name = 'states';
            
            % Define action info
            ActionInfo(1) = rlNumericSpec([2 1 1]);
            ActionInfo(1).Name = 'thrusts';
            
            % Create environment
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Update action info and initialize states and logs
            updateActionInfo(this);
            this.State = [0 this.L1 0 0 0 0]';
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0]';
            
        end
        
        function set.UseContinuousActions(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','UseContinuousActions');
            this.UseContinuousActions = logical(val);
            updateActionInfo(this);
        end
        
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',6},'','State');
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
        
        function set.ThrustLimits(this,val)
            validateattributes(val,{'numeric'},{'finite','real','vector','numel',2},'','ThrustLimits');
            this.ThrustLimits = sort(val);
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
        
        function [nextobs,reward,isdone,loggedSignals] = step(this,Action)
            
            loggedSignals = [];
            
            % Scale up the actions
            action = Action .* this.ThrustLimits(2);
            
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
            hardReset = false;
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
    
    methods (Access = private)
        
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
            this.ActionInfo(1).Name = 'thrusts';
        end
        
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
        % t     = angle of the rocket w.r.t. y-axis (rad, counterclockwise positive)
        % dx    = x velocity of rocket's center of mass (m/s)
        % dy    = y velocity of rocket's center of mass (m/s)
        % dt    = angular velocity of rocket (rad/s)
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
        %
            
            action = max(this.ThrustLimits(1),min(this.ThrustLimits(2),action));
            
            Tfwd   = action(2) + action(1);
            Ttwist = action(2) - action(1);
            
            L1_ = this.L1;
            L2_ = this.L2;
            m = this.Mass;
            g = this.Gravity;
            % inertia for a flat disk
            I = 0.5*m*L1_^2;
            
            x_  = x(1); %#ok<NASGU>
            y_  = x(2); 
            t_  = x(3);
            dx_ = x(4);
            dy_ = x(5);
            dt_ = x(6);
            
            Fx = -sin(t_) * Tfwd;
            Fy =  cos(t_) * Tfwd;
            Mz =  Ttwist * L2_  ;
            
            dx = zeros(6,1);
            
            % ground penetration
            yhat = y_ - L1_;
            
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
                dx(6) = Mz/I;
                dx(1) = dx_;
                dx(2) = dy_;
                dx(3) = dt_;
            end
            
        end
        
    end
    
end