'''
Copyright 2021  Douglas Feitosa Tomé

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

matplotlib.use('Agg')

class DataTester:
    
    
    def __init__(self, datadir=None, filename=None, columns=None):
        '''
        Provide either datadir and filename or columns for initialization.
        
        datadir: directory to csv file with tabular data
        filename: name of csv file with tabular data
        columns: list of column names
        '''

        if type(datadir) != type(None) and type(filename) != type(None):
            self.load_data_frame(datadir, filename)
        
        elif type(columns) != type(None):
            self.columns = []
            for col in columns:
                self.columns.append(col)
                setattr(self, col, [])
                
        else:
            sys.exit("Error: Provide either datadir and filename or columns for initialization.")
            

                
    def add_data(self, data):
        '''
        data: list of (column, value) tuples. values for all columns must be provided.
        '''

        col_check = 0
        for col, val in data:
            if col in self.columns:
                col_check += 1
            else:
                sys.exit("Error: attempted to add data with unknown column '" + col + "'")
            getattr(self, col).append(val)
        if col_check != len(self.columns):
            sys.exit("Error: attempted to add data with missing columns")


    def build_data_frame(self):
        data = {}
        for col in self.columns:
            data[col] = getattr(self, col)
        self.df = pd.DataFrame(data)

    
    def save_data_frame(self, datadir, filename):
        self.df.to_csv(os.path.join(datadir, filename))


    def load_data_frame(self, datadir, filename):
        self.df = pd.read_csv(os.path.join(datadir, filename))


    def select_rows(self, *args):
        '''
        args: list of (column, values) tuples. values for all columns must be provided.
              values is a tuple of the form (value1, value2, ...).
              if len(values) == 1, values is of the form (value,)
        '''
        
        select_rows = np.full((len(self.df)), True)
        '''
        print('select_rows', type(select_rows), select_rows.shape)
        print(select_rows)
        print('args', type(args))  
        print(args)
        '''
        for col,values in args:
            '''print ('col,value', col, value)'''
            select = np.full((len(self.df)), False)
            for value in values:
                select = select | (self.df[col] == value)
            select_rows = select_rows & select
        return self.df[select_rows]


    def get_data(self, data_column, *args):
        '''
        Retrieve data in specified column and rows.

        data_column: column containing desired data
        args: as in select_rows
        '''

        return self.select_rows(*args)[data_column].to_numpy()

    
    def store_data(self, data_name, data_column, *args):
        '''
        Store data in specified column and rows as a new object property.
        
        data_name: name of data store
        data_column: column containing desired data
        args: as in select_rows
        '''

        setattr(self, data_name, self.get_data(data_column, *args))
        
        data = getattr(self, data_name)
        print(data_name, type(data), data.shape)
        print(data)


    def shapiro(self, data_name):
        '''
        Perform the Shapiro-Wilk test for normality.

        null hypothesis: data was drawn from a normal distribution.
        '''
        
        return stats.shapiro(getattr(self, data_name))


    def normaltest(self, data_name):
        '''
        Test whether a sample differs from a normal distribution.

        null hypothesis: data was drawn from a normal distribution.
        '''

        data = getattr(self, data_name)
        if len(data) >= 8:
            return stats.normaltest(data)
        else:
            print("normaltest only valid with at least 8 samples. %d were given."%(len(data)))
            return None

    
    def levene(self, data1_name, data2_name):
        '''
        Perform Levene test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.levene(getattr(self, data1_name), getattr(self, data2_name))


    def bartlett(self, data1_name, data2_name):
        '''
        Perform Bartlett’s test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.bartlett(getattr(self, data1_name), getattr(self, data2_name))


    def unpaired_t(self, data1_name, data2_name, equal_var=True, alternative='two-sided'):
        '''
        Perform t-test for the means of two independent samples.

        null hypothesis: two independent samples have identical average (expected) values.
        '''
        
        return stats.ttest_ind(getattr(self, data1_name), getattr(self, data2_name), equal_var=equal_var, alternative=alternative)


    def one_samp_t(self, data1_name, popmean, alternative='two-sided'):
        '''
        Perform t-test for the mean of ONE group of scores.

        null hypothesis: mean of a sample of independent observations a is equal to the given 
                         population mean, popmean.
        '''
        
        return stats.ttest_1samp(getattr(self, data1_name), popmean, alternative=alternative)
    

    def mannwhitneyu(self, data1_name, data2_name, alternative='two-sided'):
        '''
        Perform the Mann-Whitney U rank test on two independent samples.

        null hypothesis: two independent samples x have the same underlying distribution.
        '''

        return stats.mannwhitneyu(getattr(self, data1_name), getattr(self, data2_name), alternative=alternative)



    def paired_t(self, data1_name, data2_name, alternative='two-sided'):
        '''
        Perform t-test for the means of two related samples.

        null hypothesis: two related samples have identical average (expected) values.
        '''
        
        return stats.ttest_rel(getattr(self, data1_name), getattr(self, data2_name), alternative=alternative)


    
    def wilcoxon(self, data1_name, data2_name, alternative='two-sided'):
        '''
        Perform Wilcoxon signed-rank test for the means of two related samples.

        null hypothesis: two related samples have the same underlying distribution.
        '''
        
        return stats.wilcoxon(getattr(self, data1_name), y=getattr(self, data2_name), alternative=alternative)
    

    
    def plot_line(self, x, y, hue, style, ci, datadir, filename, xlabel, ylabel, xmult, ymult, xlim, ylim, *args):
        '''
        args: as in select_rows(*args)
        '''
        
        data = self.select_rows(*args)

        data[x] = data[x] * xmult
        data[y] = data[y] * ymult
        
        '''
        print("\nFiltered data frame for plot %s:"%filename)
        print(data)
        '''
        
        plt.rcParams.update({'font.size': 8}) # 6
        plt.rcParams.update({'font.family': 'sans-serif'})
        plt.rcParams.update({'font.sans-serif': 'Verdana'})

        #fig = plt.figure(dpi=300)
        #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
        fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
        #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
  
        #palette = sns.color_palette("muted", n_colors=4)
        #palette = sns.color_palette(['#4878d0', '#ee854a', '#6acc64', '#8c613c'])
        #palette = sns.color_palette(['sandybrown', 'darkseagreen', 'thistle', 'royalblue']) # p1
        #palette = sns.color_palette(['#20639b', '#ed553b', '#f6d55c']) # p2
        #palette = sns.color_palette(['magenta']) # training
        palette = sns.color_palette(['#ed553b', '#20639b']) # training and novel recall
        #palette = sns.color_palette(['thistle']) # # discrimination
        #palette = sns.color_palette(['lightcoral', 'sandybrown', 'darkseagreen',]) # labelling
        
        
        
        ax = None
        if type(style) != type(None):
            ax= sns.lineplot(x=x, y=y, hue=hue, style=style, ci=ci, data=data, palette=palette)
        else:
            ax = sns.lineplot(x=x, y=y, hue=hue, ci=ci, data=data, palette=palette)
   
        #ax.get_legend().remove()
  
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
        #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        ax.set_xticks(list(set(data[x])))
  
        #plt.title("")

        if type(xlim) != type(None):
            plt.xlim(xlim)
        if type(ylim) != type(None):
            plt.ylim(ylim)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
        print('Saved line plot', filename)
        plt.close(fig)


def main(argv):
    # Parameters
    datadir = "~/projects/systems-consolidation/simulations/sim_rc_p11/run-001"
    filename = 'metrics-recall-all.csv'
    
    datadir = os.path.expanduser(datadir)
    datatester = DataTester(datadir=datadir, filename=filename)
    
    # Retrieve data for statistical tests
    data1 = 'data1'
    data2 = 'data2'
    
    datatester.store_data('data1',
                          'score',
                          ('region', ('hpc',)),
                          ('neuron_type',('exc',)),
                          ('learn_time', (1800,)),
                          ('consolidation_time', (86400,)),
                          ('phase', ('test',)),
                          ('pattern',('all',)),
                          ('metric',('accuracy',)))

    datatester.store_data('data2',
                          'score',
                          ('region', ('ctx',)),
                          ('neuron_type',('exc',)),
                          ('learn_time', (1800,)),
                          ('consolidation_time', (86400,)),
                          ('phase', ('test',)),
                          ('pattern',('all',)),
                          ('metric',('accuracy',)))    

    
    #'''
    # Shapiro-Wilk test for normality
    shapiro1 = datatester.shapiro('data1')
    print('\n\ndata1 shapiro')
    print(shapiro1)

    if type(data2) != type(None):
        shapiro2 = datatester.shapiro('data2')
        print('data2 shapiro')
        print(shapiro2)
    
    # D’Agostino and Pearson’s test for normality
    normaltest1 = datatester.normaltest('data1')
    print('\n\ndata1 normaltest')
    print(normaltest1)

    if type(data2) != type(None):
        normaltest2 = datatester.normaltest('data2')
        print('data2 normaltest')
        print(normaltest2)
    
    # Levene's test for equality of variances
    if type(data2) != type(None):
        levene = datatester.levene('data1', 'data2')
        print('\n\nlevene')
        print(levene)

    # Bartlett's test for equality of variances
    if type(data2) != type(None):
        bartlett = datatester.bartlett('data1', 'data2')
        print('\n\nbartlett')
        print(bartlett)
    #'''
    
    '''
    # One-sample t-test for equality of means: one independent sample and a given population mean
    if type(data2) == type(None):
        one_samp_t = datatester.one_samp_t('data1', 0, alternative='two-sided')
        print('\n\none_samp t-test')
        print(one_samp_t)
    '''

    #'''
    # Unpaired t-test for equality of means: independent samples with the same variance
    if type(data2) != type(None):
        unpaired_t = datatester.unpaired_t('data1', 'data2', equal_var=True, alternative='two-sided')
        print('\n\nunpaired t-test')
        print(unpaired_t)

    # Welch's unpaired t-test: independent samples with different variances
    if type(data2) != type(None):
        welch_unpaired_t = datatester.unpaired_t('data1', 'data2', equal_var=False, alternative='two-sided')
        print('\n\nwelch_unpaired t-test')
        print(welch_unpaired_t)

    # Mann-Whitney U test: indepent samples whose underlying distributions are not normal
    if type(data2) != type(None):
        mannwhitneyu = datatester.mannwhitneyu('data1', 'data2', alternative='two-sided')
        print('\n\nmannwhitneyu')
        print(mannwhitneyu)

    #'''

    '''
    # Paired t-test for equality of means: related samples with the same variance
    if type(data2) != type(None):
        paired_t = datatester.paired_t('data1', 'data2', alternative='two-sided')
        print('\n\npaired t-test')
        print(paired_t)

    # Wilcoxon signed-rank test: paired samples whose underlying distributions are not normal
    if type(data2) != type(None):
        wilcoxon = datatester.wilcoxon('data1', 'data2', alternative='two-sided')
        print('\n\nwilcoxon t-test')
        print(wilcoxon)
    
    '''
    
if __name__ == "__main__":
 main(sys.argv[1:])
