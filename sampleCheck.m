clear all;
A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\sample_clear.csv",0,0);
data_u = zeros(5,8192);
classNum = 5;
sampleNum = 300;
testNum = 0;

for j = 1:5
    for i = 1:8192
        data_u(j,i) = mean(A( sampleNum*(j-1)+1 : sampleNum*j, i));
    end
end
data_o = zeros(sampleNum*classNum);
for j = 1:sampleNum*classNum
    dd = 0;
    for i = 1:8192
        dd = dd + abs((A(j,i)-data_u(ceil(j/sampleNum),i)));
    end
    data_o(j) = dd/8192;
end

lossList = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\lossList.csv",0,0);
train_o = [data_o(1:sampleNum-testNum,1); data_o(sampleNum+1:sampleNum*2-testNum,1); data_o(sampleNum*2+1:sampleNum*3-testNum,1); data_o(sampleNum*3+1:sampleNum*4-testNum,1); data_o(sampleNum*4+1:sampleNum*5-testNum,1)];
    subplot(4,1,1);
for i = 1:5
    plot(train_o);
    grid on;
    set(gca,'xtick',[1,sampleNum-testNum,(sampleNum-testNum)*2,(sampleNum-testNum)*3,(sampleNum-testNum)*4])
    title('Frequency Domain')
    ylabel('MAE')
    xlabel('iPhone-7P                iPhone-8P                 Samsung                    LG                    iPhone-XR');
end
    subplot(4,1,4);
for i = 1:5
    plot(lossList(i,:));
    grid on;
    set(gca,'xtick',[1,sampleNum-testNum,(sampleNum-testNum)*2,(sampleNum-testNum)*3,(sampleNum-testNum)*4])
    title('Train set Loss')
    ylabel('loss')
    xlabel('iPhone-7P                iPhone-8P                 Samsung                    LG                    iPhone-XR');
    hold on
end
    legend('iPhone-7P', 'iPhone-8P', 'Samsung', 'LG', 'iPhone-XR')

%%% TIME DOMAIN %%%
% A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\Sample\sample_clear_time.csv",0,0,[0 0 19999999 0]);
% data = zeros(5,sampleNum,16384);
% for i = 1:5
%     cc = A((i-1)*4000000+1:i*4000000);
%     for j = 1:sampleNum
%         data(i,j,:) = cc((j-1)*16384+1:j*16384);
%     end
% end
A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\sample_clear_time.csv",0,0);
data = A;

data_time_u = zeros(5,16384);
u = zeros(16384);
for i = 1:5
    for j = 1:16384
        data_time_u(i,j) = mean(data((i-1)*sampleNum+1:i*sampleNum,j));
    end
end

data_time_o = zeros(sampleNum*classNum);
for i = 1:5
    for j = 1:sampleNum
        mae = mean(abs(data((i-1)*sampleNum+j,:)-data_time_u(i,:)));
        data_time_o((i-1)*sampleNum+j) = mae;
    end
end
train_data_time_o = [data_time_o(1:sampleNum-testNum,1); data_time_o(sampleNum+1:sampleNum*2-testNum,1); data_time_o(sampleNum*2+1:sampleNum*3-testNum,1); data_time_o(sampleNum*3+1:sampleNum*4-testNum,1); data_time_o(sampleNum*4+1:sampleNum*5-testNum,1)];
subplot(4,1,2);

plot(train_data_time_o);
grid on;
set(gca,'xtick',[1,sampleNum-testNum,(sampleNum-testNum)*2,(sampleNum-testNum)*3,(sampleNum-testNum)*4])
title('Time Domain');
xlabel('Sample');
ylabel('MAE');

%%% MIN MAX%%%
min_max = zeros(sampleNum*classNum);
for i = 1:5
    for j = 1:sampleNum
        min_max((i-1)*sampleNum+j) = max(data((i-1)*sampleNum+j,:)) - min(data((i-1)*sampleNum+j,:));
    end
end
train_minmax_time_o = [min_max(1:sampleNum-testNum,1); min_max(sampleNum+1:sampleNum*2-testNum,1); min_max(sampleNum*2+1:sampleNum*3-testNum,1); min_max(sampleNum*3+1:sampleNum*4-testNum,1); min_max(sampleNum*4+1:sampleNum*5-testNum,1)];
subplot(4,1,3);
plot(train_minmax_time_o)
title('Max-Min');
xlabel('Count');
ylabel('Max-Min');

%%%plot histogram dirgram%%%
% histogram(min_max(:,1));
% title('Max-Min Histogram');
% xlabel('Count');
% ylabel('amount');

