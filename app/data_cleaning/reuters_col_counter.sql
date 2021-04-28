do
$$
declare
  cols text;

begin

    cols := string_agg('count(' || column_name::text || ') '  || column_name::text, ',')
    from (select column_name
          from information_schema.columns
          where table_name = 'data_statements' AND table_schema = 'wrds') c;

  execute format('create view wrds.col_counter as select %s from wrds.data_statements;', cols);

end;
$$;

select * from wrds.col_counter;