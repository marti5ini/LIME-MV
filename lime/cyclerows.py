def cycle_rows(n, row, explainer, comb, config, path, start)
    for n, row in enumerate(test_dataset):
        step = time.time()
        exp = explainer.explain_instance(row, predict_fn, model_regressor=comb['model_regressor'])
        for key in exp.local_exp:
            # Use list comprehension to extract the second value of each tuple
            values = [round(x[1],5) for x in sorted(exp.local_exp[key], key=lambda x: x[0])]
            ###################################################### This can be done outside.
            cfg_idx = [0]*len(values)
            for i, idx in enumerate(cfg_idx):
                if i in config['idx']:
                    cfg_idx[i] = round(config['perc'][config['idx'].index(i)], 2)
            info = [cfg_idx, str(pl[0])+str(pl[1]), n, comb['name']]
            ######################################################
        to_append = values+info
        with open(path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter="|")
            writer.writerow(to_append)
        print('Row', n, '-', row, 'explained in:', round(time.time()-step, 4))
        print('Seconds passed:', round(time.time()-start, 4), ' \n')