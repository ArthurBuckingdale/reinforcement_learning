classdef RocketLanderVisualizer < rl.env.viz.AbstractFigureVisualizer
% ROCKETLANDERVISUALIZER Visualizer for RocketLander environment

% Copyright 2019 The MathWorks, Inc.

    methods
        function this = RocketLanderVisualizer(env)
            this = this@rl.env.viz.AbstractFigureVisualizer(env);
        end
    end
    methods (Access = protected)
        
        function f = buildFigure(this)
            f = figure('Toolbar','none',...
                'Visible','on',...
                'HandleVisibility','off', ...
                'NumberTitle','off',...
                'Name','Rocket Lander',... 
                'CloseRequestFcn',@(~,~)delete(this));
            if ~strcmp(f.WindowStyle,'docked')
                f.Position = [200 100 800 500];
            end
            % Turn the menubar off here instead of during construction to
            % run the animation while running from live script
            f.MenuBar = 'none';
            
            ha = gca(f);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            ha.ZLimMode = 'manual';
            ha.DataAspectRatioMode = 'manual';
            ha.PlotBoxAspectRatioMode = 'manual';
            ha.XLim = [-1 1]*100;
            ha.YLim = [-10 120];
            hold(ha,'on');
        end
        
        function updatePlot(this)

            env = this.Environment;
                
            f = this.Figure;
            ha = gca(f);
            
            % extract state and length props
            action = env.LastAction;
            state = env.State;
            L1 = env.L1;
            L2 = env.L2;
            thrustLims = env.Angle; %#ok<NASGU>
            l_ = L2*0.5;
            x = state(1);
            y = state(2);
            t = state(3);
            dx = state(4);
            dy = state(5);
            
            collision = (y - L1) < 0;
            roughCollision = collision && (dy < -0.5 || abs(dx) > 0.5);
            
            if collision
                y = L1;
            end
            
            % normalize action to 0-1
            %action = (action - thrustLims(1))/diff(thrustLims);
            
            c = cos(t); s = sin(t);
            R = [c,-s;s,c];
            
            bodyplot = findobj(ha,'Tag','bodyplot');
            groundplot = findobj(ha,'Tag','groundplot'); %#ok<NASGU>
            ltthrusterbaseplot = findobj(ha,'Tag','ltthrusterbaseplot');
            rtthrusterbaseplot = findobj(ha,'Tag','rtthrusterbaseplot');
            
            ltthrusterplot = findobj(ha,'Tag','ltthrusterplot');
            rtthrusterplot = findobj(ha,'Tag','rtthrusterplot');
            
            textplot = findobj(ha,'Tag','textplot');
            
            if isempty(bodyplot) || ~isvalid(bodyplot) || ...
                    isempty(ltthrusterbaseplot) || ~isvalid(ltthrusterbaseplot) || ...
                    isempty(rtthrusterbaseplot) || ~isvalid(rtthrusterbaseplot) || ...
                    isempty(ltthrusterplot) || ~isvalid(ltthrusterplot) || ...
                    isempty(rtthrusterplot) || ~isvalid(rtthrusterplot) || ...
                    isempty(textplot) || ~isvalid(textplot)
                    
                bodyplot = rectangle(ha,'Position',[x-L1 y-L1 2*L1 2*L1],...
                    'Curvature',[1 1],'FaceColor','y','Tag','bodyplot');
                groundplot = line(ha,ha.XLim,[0 0],'LineWidth',2,'Color','k','Tag','groundplot'); %#ok<NASGU>
                ltthrusterbaseplot = line(ha,[0 0],[0 0],'LineWidth',1,'Color','k','Tag','ltthrusterbaseplot');
                rtthrusterbaseplot = line(ha,[0 0],[0 0],'LineWidth',1,'Color','k','Tag','rtthrusterbaseplot');
                
                ltthrusterplot = patch(ha,[0 0 0],[0 0 0],'r','Tag','ltthrusterplot');
                rtthrusterplot = patch(ha,[0 0 0],[0 0 0],'r','Tag','rtthrusterplot');
                
                textplot = text(ha,0,0,'','Color','r','Tag','textplot');
            end
            
            bodyplot.Position = [x-L1 y-L1 2*L1 2*L1];
            
            LL1 = [-L2-l_;0];
            LL2 = [-L2+l_;0];
            LR1 = [+L2-l_;0];
            LR2 = [+L2+l_;0];
            
            TL1 = [-L2-l_;0];
            TL2 = [-L2+l_;0];
            TL3 = [-L2   ;-action(1)*10];
            TR1 = [+L2-l_;0];
            TR2 = [+L2+l_;0];
            TR3 = [+L2   ;-action(2)*10];
            
            in = [LL1 LL2 LR1 LR2 TL1 TL2 TL3 TR1 TR2 TR3];
            out = R*in + [x;y];
            
            ltthrusterbaseplot.XData = out(1,1:2);
            ltthrusterbaseplot.YData = out(2,1:2);
            rtthrusterbaseplot.XData = out(1,3:4);
            rtthrusterbaseplot.YData = out(2,3:4);
            
            ltthrusterplot.XData = out(1,5:7 );
            ltthrusterplot.YData = out(2,5:7 );
            rtthrusterplot.XData = out(1,8:10);
            rtthrusterplot.YData = out(2,8:10);
            
            if roughCollision
                textplot.String = 'Ouch!!!';
                textplot.String = num2str(dy);
                textplot.Position = [x,-5];
            else
                textplot.String = '';
            end
            
            % Refresh rendering in figure window
            drawnow();
        end
    end
end