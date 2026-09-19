
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(plyr)
library(maps)
library(mapproj)

hep <- read.csv("../input/hepatitis.csv")
meas <- read.csv("../input/measles.csv")
mump <- read.csv("../input/mumps.csv")
polio <- read.csv("../input/polio.csv")
pert <- read.csv("../input/pertussis.csv")
rub <- read.csv("../input/rubella.csv")
smlpx <- read.csv("../input/smallpox.csv")

all <- rbind(hep, meas, mump, pert, polio, rub, smlpx)

dat <- ddply(all, c("state", "disease"), function(x){
  mean.inc <- mean(x$incidence_per_capita)
  data.frame(mean_inc_perCap = mean.inc)
})

attach(dat)
gg<-ggplot(dat, aes(state, mean_inc_perCap))
gg + geom_bar(position = "stack", stat = "identity", aes(fill = disease))+
  theme_minimal()+scale_fill_brewer(palette = "Paired")+ggtitle("Mean incidence per capita by state, all years")
  
  
  
measles<-subset(dat, dat$disease=="MEASLES")
measles$state_name<-all$state_name[match(measles$state, all$state)]
measles$region <- tolower(measles$state_name)
states <- map_data("state")
map.df<-merge(states, measles, by = "region", all.x=T)
map.df <- map.df[order(map.df$order),]
ggplot(map.df, aes(x=long,y=lat,group=group))+
  geom_polygon(aes(fill=mean_inc_perCap))+
  geom_path()+ ggtitle("Measles in the US")+
  scale_fill_gradientn(colours=rev(heat.colors(10)),na.value="grey90")+
  coord_map()+theme_minimal()
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(plyr)
library(maps)
library(mapproj)

hep <- read.csv("../input/hepatitis.csv")
meas <- read.csv("../input/measles.csv")
mump <- read.csv("../input/mumps.csv")
polio <- read.csv("../input/polio.csv")
pert <- read.csv("../input/pertussis.csv")
rub <- read.csv("../input/rubella.csv")
smlpx <- read.csv("../input/smallpox.csv")

all <- rbind(hep, meas, mump, pert, polio, rub, smlpx)

dat <- ddply(all, c("state", "disease"), function(x){
  mean.inc <- mean(x$incidence_per_capita)
  data.frame(mean_inc_perCap = mean.inc)
})

attach(dat)
gg<-ggplot(dat, aes(state, mean_inc_perCap))
gg + geom_bar(position = "stack", stat = "identity", aes(fill = disease))+
  theme_minimal()+scale_fill_brewer(palette = "Paired")+ggtitle("Mean incidence per capita by state, all years")
  
  
  
measles<-subset(dat, dat$disease=="MEASLES")
measles$state_name<-all$state_name[match(measles$state, all$state)]
measles$region <- tolower(measles$state_name)
states <- map_data("state")
map.df<-merge(states, measles, by = "region", all.x=T)
map.df <- map.df[order(map.df$order),]
ggplot(map.df, aes(x=long,y=lat,group=group))+
  geom_polygon(aes(fill=mean_inc_perCap))+
  geom_path()+ ggtitle("Measles in the US")+
  scale_fill_gradientn(colours=rev(heat.colors(10)),na.value="grey90")+
  coord_map()+theme_minimal()