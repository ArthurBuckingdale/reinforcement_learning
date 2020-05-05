function [dx,Tfwd,Ttwist,action] = dynamics_rocket_lander(this,x,action)
            %this is a direct copy of the dynamics function from the class.             
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
            
            %Tupwards
            Tupwards   =   this.Thrust.*cosd(action(2)-60) + this.Thrust.*cosd(action(1)-60)+...
                this.Thrust.*cosd(action(3)-60) + this.Thrust.*cosd(action(4)-60);
            %             Tforwards   =   this.Thrust.*cosd(action(2)) + this.Thrust.*cosd(action(1));
            %             Trightwards = this.Thrust.*cosd(action(2)) + this.Thrust.*cosd(action(1));
            %this is the left and right tilting motion
            Ttheta = this.Thrust.*sind(action(1)-60).^2 + this.Thrust.*sind(action(3)-60).^2 ...
                - this.Thrust.*sind(action(2)-60).^2 - this.Thrust.*sind(action(4)-60).^2;
            Ttheta2 = this.Thrust.*cosd(action(1)-60).^2 + this.Thrust.*cosd(action(3)-60).^2 ...
                - this.Thrust.*cosd(action(2)-60).^2 - this.Thrust.*cosd(action(4)-60).^2;
            %this is the yaw motion
            Tphi = sind(action(5)-30)*cosd(action(5)-30)*(this.FinArea)*(1.225)*(0.8)*(x(5).^2);  %need to decide plus and minus here
            %so this it the forwards and reverse tilting action
            Trho = this.Thrust.*sind(action(2)-60).^2 + this.Thrust.*sind(action(1)-60).^2 ...
                - this.Thrust.*sind(action(3)-60).^2 - this.Thrust.*sind(action(4)-60).^2;
            Trho2 = this.Thrust.*cosd(action(2)-60).^2 + this.Thrust.*cosd(action(1)-60).^2 ...
                - this.Thrust.*cosd(action(3)-60).^2 - this.Thrust.*cosd(action(4)-60).^2;
            
            Tfwd=1;
            Ttwist=1;
            
            
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
            I1 = 0.5*m*L1_^2; % only half of the mass is inertial
            I2 = 0.5*m*L2_^2; % only half of the mass is inertial
            I3 = 0.5*m*L3_^2; % only half of the mass is inertial
            
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
            Fx = -sin(th_)*(cos(ph_).^2)*Tupwards-sin(rh_)*(sin(ph_).^2)*Tupwards; %pushing body left and right
            Fy =  cos(th_)*cos(rh_)* Tupwards; %pushing the body upwards
            Fz = -sin(rh_)*(cos(ph_).^2)*Tupwards-sin(th_)*(sin(ph_).^2)*Tupwards ; %pushing body in and out of screen.
            My =  Tphi * sqrt(L2_.^2+L3_.^2);  %rotation about the y-axis is yawing L2 is the rocket radius.(I know this is no correct)
            Mx =  (Trho * L3_) + (Trho2 * L1_);              %rotation about the x-axis is the inward and outward          
            Mz =  (Ttheta * L2_) + (Ttheta2 * L1_);               %rotation about the z-axis is left and right   
            
            dx = zeros(12,1);
            
            % below are the velocities and accelerations which are caused
            % by the forces here. I'm assuming perfect rotational symmetry
            % for the thing, even though it's not the case
            % treat as "falling" mass
            dx(4) = Fx/m;   %add some jitters here in future. 
            dx(5) = Fy/m - g; %gravity is in this direction
            dx(6) = Fz/m;  %we can maybe add some random jitters here in future
            dx(10)= Mz/I1;
            dx(11)= My/I2;
            dx(12)= Mx/I1; 
            dx(1) = dx_;
            dx(2) = dy_;
            dx(3) = dz_;
            dx(7) = dth_;
            dx(8) = dph_;
            dx(9) = drh_;
           
        end