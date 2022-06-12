# Motionful API by PHP with Laravel Framework

Developed by the Bangkit 2022 Cohort Team Cloud Computing Division

# API Documentation

### User API registration
parameters that must be met

+ name
+ email
+ password 
+ password_confirmation

example by url method Post

```code
http://YourHost:8000/api/register?name=Motionful&email=motiontest@gmail.com&password=test12345&password_confirmation=test12345
```
example response  if account created
```json
{
    "success": true,
    "user": {
        "name": "Motionful",
        "email": "motiontest@gmail.com",
        "updated_at": "2022-06-08T11:26:15.000000Z",
        "created_at": "2022-06-08T11:26:15.000000Z",
        "id": 1
    }
}
```

### Login API
C'soon

### Upload API
C'soon


## Deploy to server

Deploy by VM Ubuntu Server

NB: If you are running this server on Google Cloud, you need to allow port 8000.

### Install apache
Step #1
```bash
sudo apt-get update
```
Step #2
```bash
sudo apt-get install apache2 libapache2-mod-php
```




### Install MySQL

```bash
sudo apt-get install mysql-server php-mysql
```



### Install php

```bash
sudo apt-get install php libapache2-mod-php php-common php-mbstring php-xmlrpc php-soap php-gd php-xml php-mysql php-cli php-mcrypt php-zip
```

### goto data


```bash
cd /var/www/html/
```

to make sure php is running you cant use this simple code to checking php version 

```bash
cat test.php
```

```php
<?php 
phpinfo(); 
?>
```
upload a source code to this folder

configuration in .env

### Run API server
```php
php artisan serve
```

## 
