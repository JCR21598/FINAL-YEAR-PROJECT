
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
    console.log($('#detector-field').val())
    $.ajax({
        url: "/input/",
        type: "POST",
        data: {
            url_input: $('#detector-field').val()
            },

        // Message before sending
       beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
                }
        },

        // If AJAX successful
       success: function(json_response){
            alert("Successful AJAX")
            $('#detector-field').val('');  // Reset TextField



            // Changes HTML elements
            $('.resp-url').html(json_response.url);
            $('.resp-news-title').html(json_response.article_title);
            $('.resp-news-text').html(json_response.article_text);
        },

        // If AJAX unsuccessful
        error: function(){
            alert("AJAX Error")
        }
    });
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