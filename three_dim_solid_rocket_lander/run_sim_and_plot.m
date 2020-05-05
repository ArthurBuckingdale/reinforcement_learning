function run_sim_and_plot(agent,env)
%the purpose of this script is for post processing simulation data. Since I
%don't want to desing an animation for each different RL agent that I
%program, it's important to be able to plot the states and observe what
%cause the program to end. The data is stored in a timeseries, even though
%one of the dimensions is unused, it bothers me a little. Anyway, this
%function will run a simulation and return the plots for each entry of the
%vector. This will allow me to figure out what the agent is doing to debug
%the physics section of the problem.

close all

%% run a simulation
%in this script, we're not going to do a statistical analysis of what
%primarily causes the system to end an episode, but the charts of the data
%progression.

results=sim(agent,env);

%% obtain the observation information
%these are the various states that the agent is moving through, it is the
%position and velocity for all things. We just need to reshape as it is not
%well laid out.

data=results.Observation.states.Data;
[vv,~,nn]=size(data);
data=reshape(data,vv,nn);

%% obtain the action info

action=results.Action.angles.Data;
[gg,~,hh]=size(action);
action=reshape(action,gg,hh);

%% plot the translational state vectors
%so the first plot will contain the translational positions and the second
%will contain the translational velocities

%translational positions
subplot(2,4,1)
hold on
plot(data(1,2:end),'b*')
plot(data(2,2:end),'r*')
plot(data(3,2:end),'g*')
title('Translational Positions as a Function of Time')
xlabel('Time(each index is 0.05s)')
ylabel('Position')
legend('X pos','Y pos','Z pos')
hold off

%translational velocities
subplot(2,4,2)
hold on
plot(data(4,2:end),'b*')
plot(data(5,2:end),'r*')
plot(data(6,2:end),'g*')
title('Translational Velocities as a Function of Time')
xlabel('Time(each index is 0.05s)')
ylabel('Velocity')
legend('X vel','Y vel','Z vel')
hold off

%% plot the rotational state vectors
%we now want to see how the rotational axes are behaving in the simulation

%rotational positions
subplot(2,4,3)
hold on
plot(data(7,2:end),'b*')
plot(data(8,2:end),'r*')
plot(data(9,2:end),'g*')
title('Rotational Positions as a Function of Time')
xlabel('Time(each index is 0.05s)')
ylabel('Position')
legend('theta pos','phi pos','rho pos')
hold off

%rotational velocities
subplot(2,4,4)
hold on
plot(data(10,2:end),'b*')
plot(data(11,2:end),'r*')
plot(data(12,2:end),'g*')
title('Rotational Velocities as a Function of Time')
xlabel('Time(each index is 0.05s)')
ylabel('Velocity')
legend('theta vel','phi vel','rho vel')
hold off

%% plot the accelerations of the body 
%so this is less trivial, we need to input the action and state into our
%dynamics function to find out what the agent is doing

 bounds = [100 120-env.L1 100 60 60 60 pi pi pi pi/2 4*pi pi/2];
for i=2:hh
    [dx(i).accel,~,~,~] = dynamics_rocket_lander(env,data(:,i).*bounds,action(:,i)*(180/pi));
end

%the translational accelerations
mat=cat(2,dx.accel);
subplot(2,4,5)
hold on
plot(mat(4,:),'b*')
plot(mat(5,:),'r*')
plot(mat(6,:),'g*')
title('Translational Accel as a Function of Time')
xlabel('Time(each index is 0.05s)')
ylabel('Acceleration')
legend('X acc','Y acc','Z acc')
hold off

%the rotational accelerations
subplot(2,4,6)
hold on
plot(mat(10,:),'b*')
plot(mat(11,:),'r*')
plot(mat(12,:),'g*')
title('Rotational Accel as a Function of Time')
xlabel('Time(each index is 0.05s)')
ylabel('Acceleration')
legend('theta acc','Phi acc','Rho acc')
hold off

%% plot the actions taken by the agent
disp(action)
subplot(2,4,7)
hold on
plot(action(1,2:end),'b*')
plot(action(2,2:end),'r*')
plot(action(3,2:end),'g*')
plot(action(4,2:end),'k*')
plot(action(5,2:end),'y*')
title('Action')
legend('1','2','3','4','5')
hold off

