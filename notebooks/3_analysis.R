library(ggplot2)
library(reshape)
library(scales)
rm(list=ls())
setwd('/Users/sambarrows/Dropbox/Projects/school_websites/notebooks')

##Load lda output
lda = read.csv('../my_datasets/lda_dat.csv')
nrow(lda)

##Add variable with collapsed school types 
school_char = read.csv('../my_datasets/prepped.csv')
school_char = school_char[,c('URN', 'school_type', 'blurb')]
nrow(school_char)

##Add relevant edubase data (for now, just Phase of Education)
edubase = read.csv('../datasets/edubase_datasets/edubasealldata.csv')
edubase = edubase[,c('URN', 'PhaseOfEducation..name.')]

##Merge lda and school characteristics data
df = merge(lda, school_char, by='URN', all.x=TRUE)
nrow(df)

##Then merge with edbuase data
df = merge(df, edubase, by='URN', all.x=TRUE)
nrow(df)

names(df)[names(df)=='PhaseOfEducation..name.'] = 'Phase'

rm(school_char, edubase)

##Function for saving plots
saveplot <- function(myPlot, myPlot_name){
  pdf(paste(myPlot_name,".pdf",sep=""), width=9, height=5)
  print(myPlot)
  dev.off()
} 

#################################
##Variation across school types##
#################################

##Number obs in each phase of school
df$Phase1 = as.character(df$Phase)
df[which(df$Phase1=='Middle Deemed Secondary'),]$Phase1 = 'Middle Deemed\n Secondary'
df[which(df$Phase1=='Middle Deemed Primary'),]$Phase1 = 'Middle Deemed\n Primary'

phase_school = ggplot(df, aes(factor(Phase1))) +
  geom_bar(fill="cornflowerblue") +
  ylab("Frequency") + 
  xlab("School Phase") +
  theme_bw() + 
  theme(axis.title.x = element_text(size=12),        # can't get vjust to work
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),  
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) 
#ggsave(filename = 'hist_phase_school.pdf')   # doesn't work - text very small
saveplot(phase_school, "hist_phase_school")

##Look at either primary schools or secondary schools
prim = FALSE
if (prim==TRUE){
  df = df[which(df$Phase=='Primary'),]
  nrow(df)
} else if (prim==FALSE){
  df = df[which(df$Phase=='Secondary'),]
  nrow(df)
}

##Number obs in each (collapsed) school type
ggplot(df, aes(factor(school_type))) +
  geom_bar()

##Average proportion for each topic, by school type
school_types = unique(df$school_type)
topics = colnames(lda)[3:length(colnames(lda))]
school_topics = data.frame(matrix(NA, ncol=length(school_types), nrow=length(topics)), 
                           row.names=topics)
colnames(school_topics) = school_types
for (i in 1:length(school_types)){
  for (j in 1:length(topics)){
    school_topics[j,i] = mean(df[which(df$school_type==school_types[i]),topics[j]])
  }
}
school_topics_m = melt(cbind(school_topics, ind=rownames(school_topics)), id.vars = c('ind'))
school_topics = ggplot(school_topics_m, aes(x = variable, y = value,fill = ind)) + 
  geom_bar(position = 'fill',stat = 'identity') + 
  scale_y_continuous(labels = percent_format()) +
  ylab("Mean Topic Proportions") + 
  xlab("School Type") +
  scale_fill_discrete(name="Topics") +
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),  
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
saveplot(school_topics, "school_topics")
  
  
######################################
##Variation with test scores and fsm##
######################################

##Data from: https://www.compare-school-performance.service.gov.uk/download-data

school_var = 'ks4'

if (school_var=='ks2'){
  ##KS2 Results
  dat = read.csv('../datasets/2015-2016-england_ks2final.csv')
  dat = dat[,c('URN', 'MATPROG')]
  dat[which(dat$MATPROG=='SUPP' | dat$MATPROG=='LOWCOV'),]$MATPROG = ''
  dat$MATPROG = as.numeric(as.character(dat$MATPROG))
  dat$sch_var = dat$MATPROG
} else if(school_var=='ks4'){
  ##KS4 Results
  dat = read.csv('../datasets/2015-2016-england_ks4revised.csv')
  dat = dat[,c('URN','P8MEA')]
  dat[which(dat$P8MEA=='NP' | dat$P8MEA=='NE' | dat$P8MEA=='SUPP' | dat$P8MEA=='LOWCOV'),]$P8MEA = ''
  dat$P8MEA = as.numeric(as.character(dat$P8MEA))
  dat$sch_var = dat$P8MEA
} else if (school_var=='fsm'){
  ##Free School Meals
  dat = read.csv('../datasets/2015-2016-england_cfr.csv')
  dat = dat[,c('URN', 'FSM')]
  dat$sch_var = dat$FSM
}

df = merge(df,dat, by='URN', all.x=TRUE)
length(which(!is.na(df$sch_var)))
length(which(is.na(df$sch_var)))

#Plot distribution of school characteristic
df_mat = df[!is.na(df$sch_var),]
ggplot(df_mat, aes(df_mat$sch_var)) + 
  geom_histogram()

# Multiple plot function
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

##Plot each topic share against school characteristic
plt_offerings = ggplot(df_mat, aes(x=sch_var, y=Offerings)) + 
  geom_point(size=.8) + geom_smooth() + 
  ggtitle("Offerings") + ylab("Topic Proportion") +  xlab("") + 
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),
        plot.title = element_text(hjust = 0.5, size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
plt_happiness = ggplot(df_mat, aes(x=sch_var, y=Happiness)) + 
  geom_point(size=.8) + geom_smooth() + 
  ggtitle("Happiness") + ylab("Topic Proportion") +  xlab("") + 
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),
        plot.title = element_text(hjust = 0.5, size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
plt_inspect = ggplot(df_mat, aes(x=sch_var, y=Inspections)) + 
  geom_point(size=.8) + geom_smooth() + 
  ggtitle("Inspections") + ylab("Topic Proportion") +  xlab("Maths Progress Measure") + 
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),
        plot.title = element_text(hjust = 0.5, size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
plt_success = ggplot(df_mat, aes(x=sch_var, y=Success)) + 
  geom_point(size=.8) + geom_smooth() + 
  ggtitle("Success") + ylab("") +  xlab("") + 
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),
        plot.title = element_text(hjust = 0.5, size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
plt_support = ggplot(df_mat, aes(x=sch_var, y=Support)) + 
  geom_point(size=.8) + geom_smooth() + 
  ggtitle("Support") + ylab("") +  xlab("") + 
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),
        plot.title = element_text(hjust = 0.5, size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
plt_develop = ggplot(df_mat, aes(x=sch_var, y=Development)) + 
  geom_point(size=.8) + geom_smooth() + 
  ggtitle("Development") + ylab("") +  xlab("Maths Progress Measure") + 
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),
        plot.title = element_text(hjust = 0.5, size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
pdf("performance.pdf", width=9, height=8)
multiplot(plt_offerings, plt_happiness, plt_inspect, 
          plt_success, plt_support, plt_develop, cols=2)
dev.off()

#####################################
##Variation with Ofsted performance##
#####################################

##Ofsted data from:https://www.gov.uk/government/statistics/maintained-schools-and-academies-inspections-and-outcomes-as-at-31-august-2016
## Overall effectiveness: 1 outstanding, 2 good, 3 satisfactory, 4 inadequate
## (see http://www.education.gov.uk/schools/performance/2013/secondary_13/s13.html)

## Maintained schools and academies inspections and outcomes as at 31 August 2016:
ofsted = read.csv('../datasets/Ofsted.csv')
ofsted = ofsted[,c('URN', 'Overall.effectiveness')]
df = merge(df, ofsted, by='URN', all.x=TRUE)

df$Grade = NA
df[which(df$Overall.effectiveness==1),]$Grade = "Outstanding" 
df[which(df$Overall.effectiveness==2),]$Grade = "Good"
df[which(df$Overall.effectiveness==3),]$Grade = "Satisfactory" 
df[which(df$Overall.effectiveness==4),]$Grade = "Inadequate"
  
##Number obs getting each Ofsted grade
ofsted_grades = ggplot(df[!is.na(df$Grade),], aes(factor(Grade))) +
  geom_bar(fill="cornflowerblue") +
  ylab("Frequency") + 
  xlab("Ofsted Grade") +
  theme_bw() + 
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),  
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) 
saveplot(ofsted_grades, "ofsted_grades")

##Average proportion for each topic, by Ofsted performance
ofsted_grades = unique(df[!is.na(df$Grade),]$Grade)
topics = colnames(lda)[3:length(colnames(lda))]
school_grades = data.frame(matrix(NA, ncol=length(ofsted_grades), nrow=length(topics)), 
                           row.names=topics)
colnames(school_grades) = ofsted_grades
red_df = df[!is.na(df$Grade),]
for (i in 1:length(ofsted_grades)){
  for (j in 1:length(topics)){
    school_grades[j,i] = mean(red_df[which(red_df$Grade==ofsted_grades[i]),topics[j]])
  }
}
school_grades_m = melt(cbind(school_grades, ind=rownames(school_grades)), id.vars = c('ind'))
ofsted_topics = ggplot(school_grades_m, aes(x = variable, y = value,fill = ind)) + 
  geom_bar(position = 'fill',stat = 'identity') + 
  scale_y_continuous(labels = percent_format()) +
  ylab("Mean Topic Proportions") + 
  xlab("Ofsted Grade") +
  scale_fill_discrete(name="Topics") +
  theme_bw() +
  theme(axis.title.x = element_text(size=12),        
        axis.title.y = element_text(size=12),
        axis.text.x = element_text(size=10, colour='black'),  
        axis.text.y = element_text(size=10, colour='black'),  
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
saveplot(ofsted_topics, "ofsted_topics")

round(school_grades*100,0)

##School judged inadequate have higher proportions for discussion of success.
##Let's take a look at these examples (although few of them)
# ggplot(df, aes(df$Inspections)) + geom_histogram()
# df[which(df$Overall.effectiveness==4 &  df$Inspections>.2), 'URN']
# df[which(df$Overall.effectiveness==4 &  df$Inspections>.2), 'blurb'][1:5]

