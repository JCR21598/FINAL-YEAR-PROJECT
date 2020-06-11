

// JQuery Functionality - waits till page is loaded
$(document).ready(function(){

    //  Waits for a submit event to happen on the form
    $("#detector-form").on("submit", function(event){

        // Prevent webpage from reloading
        event.preventDefault();

        // Call AJAX function
        detector_form();
    });

    // Add more dynamic features to the website here



});

// AJAX function
function detector_form(){

    alert($('#detector-field').val())

    $.ajax({
        url: "/input/",
        type: "POST",
        data: {
            user_input: $('#detector-field').val()
            },

        // Message before sending
       beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
                }
        },

        // If AJAX successful
       success: function(view_response){
            alert("Successful AJAX")

            //  Reset elements
            $('#detector-field').val('');
            $("#detector-response").html("");

            //  Send News report for each response response
            view_response.forEach(sendToTemplate);

        },

        // If AJAX unsuccessful
        error: function(){
            alert("AJAX Error");
        }
    });
}


function sendToTemplate(each_response){

    console.log(each_response);

    //  Create div for each report
    var eachNewsDiv = document.createElement("div");
    eachNewsDiv.setAttribute("class", "each-news-response");

    //  Create necessary html elements and their attributes
    var newsUrl = document.createElement("div");
    var newsTitle = document.createElement("div");
    var newsText = document.createElement("div");
    var newsPrediction = document.createElement("div");

    newsUrl.setAttribute("class", "resp-url");
    newsTitle.setAttribute("class", "resp-news-title");
    newsText.setAttribute("class", "resp-news-text");
    newsPrediction.setAttribute("class", "resp-prediction");

    //  Add response from view to new elements
    var respUrl = document.createTextNode(each_response.url);
    var respNewsTitle = document.createTextNode(each_response.article_title);
    var respNewsText = document.createTextNode(each_response.article_text);
    var respPrediction = document.createTextNode(each_response.prediction);

    //  Link HTML elements with response content
    newsUrl.appendChild(respUrl)
    newsTitle.appendChild(respNewsTitle)
    newsText.appendChild(respNewsText)
    newsPrediction.appendChild(respPrediction)

    //  Add elements to the new div
    eachNewsDiv.appendChild(newsUrl);
    eachNewsDiv.appendChild(newsTitle);
    eachNewsDiv.appendChild(newsText);
    eachNewsDiv.appendChild(newsPrediction);

    //  Add div to the response div already created in the HTML
    document.getElementById("detector-response").appendChild(eachNewsDiv);
}






// Provided by Django
function getCookie(name) {

    var cookieValue = null;
    if (document.cookie && document.cookie != '') {

        var cookies = document.cookie.split(';');

        for (var i = 0; i < cookies.length; i++) {

            var cookie = jQuery.trim(cookies[i]);

            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {

                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');


 function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}



//$(document).on('submit', '#detector-form', function(e) {
//
//
//    e.preventDefault(); //Meant to prevent from page to reload
//
//    $.ajax({
//        type: "POST",
//        url: "/input/",
//        data: {
//            input: $("#detector-input").val()
//        },
//        success: function(data){
//            alert(data)
//        }
//    })
//});