create table tf_up as
select count(*), event_type_code, party_rk from table
group by 2,3;
create table tf_down as
select count(*),  party_rk from table
group by 2;
create table idf_up as
select count(distinct party_rk) from table;
create table idf_down as
select event_type_code, count(distinct party_rk) from table 
group by 1;
drop table if exists usr_wrk.np_nastya;

create table data as
select tf_up.party_rk, tf_up.event_type_code, tf_up.count*1.0/tf_down.count*log(idf_up.count/idf_down.count) from
tf_up join tf_down on 
tf_up.party_rk = tf_down.party_rk
join idf_up on 1=1
join idf_down on idf_down.event_type_code = tf_up.event_type_code
